from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Any, Dict, List
import yaml

from .cost_aware_scheduler import get_current_price
from .hardware_detect import (
    list_cpus,
    list_gpus,
    list_fpgas,
    list_loihi,
    list_analog,
)
from . import analog_backend

try:  # pragma: no cover - optional dependency
    import psutil
except Exception:
    psutil = None  # type: ignore

try:  # pragma: no cover - optional torch dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - allow running without torch
    class _DummyCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def memory_allocated() -> int:
            return 0

        @staticmethod
        def get_device_properties(_: int) -> Any:
            class _P:
                total_memory = 1

            return _P()

    torch = type("torch", (), {"cuda": _DummyCuda})()  # type: ignore

from .compute_budget_tracker import ComputeBudgetTracker
from .telemetry import TelemetryLogger


class AdaptiveScheduler:
    """GPU-aware scheduler with compute budget checks and carbon-aware dispatch."""

    def __init__(
        self,
        budget: ComputeBudgetTracker,
        run_id: str,
        max_mem: float = 0.9,
        check_interval: float = 1.0,
        window: int = 3,
        min_improvement: float = 0.01,
        *,
        energy_scheduler: bool = False,
        intensity_threshold: float = 0.5,
        battery_scheduler: bool = False,
        battery_threshold: float = 0.2,
        region_config: str | None = None,
    ) -> None:
        if energy_scheduler and type(self) is AdaptiveScheduler:
            from .energy_aware_scheduler import EnergyAwareScheduler
            self.__class__ = EnergyAwareScheduler
            EnergyAwareScheduler.__init__(
                self,
                budget,
                run_id,
                max_mem=max_mem,
                check_interval=check_interval,
                window=window,
                min_improvement=min_improvement,
                intensity_threshold=intensity_threshold,
            )
            return
        if battery_scheduler and type(self) is AdaptiveScheduler:
            from .battery_aware_scheduler import BatteryAwareScheduler
            self.__class__ = BatteryAwareScheduler
            BatteryAwareScheduler.__init__(
                self,
                budget,
                run_id,
                max_mem=max_mem,
                check_interval=check_interval,
                window=window,
                min_improvement=min_improvement,
                battery_threshold=battery_threshold,
            )
            return
        self.budget = budget
        self.run_id = run_id
        self.telemetry: TelemetryLogger = budget.telemetry
        self.history: Deque[float] = deque(maxlen=window)
        self.min_improvement = min_improvement
        self.max_mem = max_mem
        self.check_interval = check_interval
        self.devices: Dict[str, List[str]] = {
            "cpu": list_cpus(),
            "gpu": list_gpus(),
            "fpga": list_fpgas(),
            "loihi": list_loihi(),
            "analog": list_analog(),
        }
        self.queues: Dict[str, list[tuple[Callable[[], None], str | None]]] = {
            k: [] for k, v in self.devices.items() if v
        }
        self.region_providers: dict[str, str] = {}
        if region_config:
            try:
                with open(region_config) as f:
                    self.region_providers = yaml.safe_load(f) or {}
            except Exception:
                self.region_providers = {}
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.budget.start(run_id)
        self.thread.start()

    # --------------------------------------------------------------
    def add(
        self,
        job: Callable[[], None],
        *,
        device: str = "gpu",
        region: str | None = None,
    ) -> None:
        """Queue a job for a specific device with an optional region hint."""
        if device not in self.queues:
            self.queues[device] = []
        self.queues[device].append((job, region))

    # --------------------------------------------------------------
    def record_improvement(self, val: float) -> None:
        """Add a new improvement metric."""
        self.history.append(float(val))

    # --------------------------------------------------------------
    def _should_pause(self) -> bool:
        if self.budget.remaining(self.run_id) <= 0:
            return True
        if len(self.history) == self.history.maxlen:
            avg = sum(self.history) / len(self.history)
            if avg < self.min_improvement:
                return True
        return False

    # --------------------------------------------------------------
    def _get_cost_index(self, region: str | None) -> float:
        provider = self.region_providers.get(region or "default")
        if provider:
            price = get_current_price(provider, region or "", "m5.large")
            carbon = self.telemetry.get_live_carbon_intensity(region)
            return price * carbon
        return self.telemetry.get_cost_index(region)

    # --------------------------------------------------------------
    def _device_utilization(self, device: str) -> float:
        if device == "gpu" and torch.cuda.is_available():
            try:
                return (
                    torch.cuda.memory_allocated()
                    / torch.cuda.get_device_properties(0).total_memory
                )
            except Exception:
                return 0.0
        if device == "cpu" and psutil is not None:
            try:
                return psutil.cpu_percent(interval=None) / 100.0
            except Exception:
                return 0.0
        if device == "analog" and getattr(analog_backend, "_HAS_ANALOG", False):
            sim = getattr(analog_backend, "analogsim", None)
            if sim is not None and hasattr(sim, "utilization"):
                try:
                    util = sim.utilization()
                    return float(util)
                except Exception:
                    return 0.0
        return 0.0

    # --------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            if not any(self.queues.values()):
                time.sleep(self.check_interval)
                continue
            if self._should_pause():
                time.sleep(self.check_interval)
                continue
            candidates: list[tuple[float, str, int, Callable[[], None]]] = []
            for dev, queue in self.queues.items():
                if queue and self._device_utilization(dev) < self.max_mem:
                    for i, (job, region) in enumerate(queue):
                        cost = self._get_cost_index(region)
                        candidates.append((cost, dev, i, job))
            if candidates:
                cost, dev, idx, job = min(candidates, key=lambda x: x[0])
                self.queues[dev].pop(idx)
                res = job()
                if isinstance(res, (int, float)):
                    self.record_improvement(float(res))
            else:
                time.sleep(self.check_interval)
        self.budget.stop()

    # --------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.budget.stop()

    # --------------------------------------------------------------
    def report_load(self) -> dict[str, float]:
        stats = self.telemetry.get_stats()
        stats["available_devices"] = {k: len(v) for k, v in self.devices.items()}
        return stats


__all__ = ["AdaptiveScheduler"]

