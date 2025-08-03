"""Consolidated scheduler implementations."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path
import sys

from .scheduler_utils import (
    cpu_utilization,
    gpu_utilization,
    tpu_utilization,
    analog_utilization,
    battery_level,
)

pkg = sys.modules.get('asi')
if pkg is not None and not getattr(pkg, '__path__', None):
    pkg.__path__ = [str(Path(__file__).parent)]

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

try:  # pragma: no cover - optional dependency
    from .carbon_aware_scheduler import get_current_price  # type: ignore
except Exception:  # pragma: no cover - fallback when stub not set
    def get_current_price(*_a, **_k) -> float:
        return 0.0

# ``hardware_detect`` may rely on optional backends, so import lazily
list_cpus = list_gpus = list_fpgas = list_loihi = list_analog = None
analog_backend = None
from .compute_budget_tracker import ComputeBudgetTracker
from .telemetry import TelemetryLogger


@dataclass
class BudgetAwareScheduler:
    """Reduce batch size and learning rate as the budget shrinks."""

    tracker: ComputeBudgetTracker
    run_id: str = "default"
    threshold: float = 1.0

    def schedule_step(self, config: Any) -> None:
        """Modify ``config`` in-place if budget is below ``threshold``."""
        if self.tracker.remaining(self.run_id) < self.threshold:
            if hasattr(config, "batch_size"):
                config.batch_size = max(1, config.batch_size // 2)
            if hasattr(config, "lr"):
                config.lr *= 0.5


class AcceleratorScheduler:
    """Queue jobs and execute when device utilization is low."""

    def __init__(
        self,
        max_util: float = 0.9,
        check_interval: float = 1.0,
        *,
        max_temp: Optional[float] = None,
        telemetry: Optional[TelemetryLogger] = None,
    ) -> None:
        self.max_util = max_util
        self.check_interval = check_interval
        self.max_temp = max_temp
        if telemetry is not None:
            self.telemetry = telemetry
        elif max_temp is not None:
            from .telemetry import TelemetryLogger as _TelemetryLogger

            self.telemetry = _TelemetryLogger(interval=check_interval)
        else:
            self.telemetry = None
        self.queues: Dict[str, Deque[Callable[[], None]]] = {
            "cpu": deque(),
            "gpu": deque(),
            "tpu": deque(),
        }
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # --------------------------------------------------------------
    def add(self, job: Callable[[], None], accelerator: str = "gpu") -> None:
        self.queues.setdefault(accelerator, deque()).append(job)

    # --------------------------------------------------------------
    def _utilization(self, accelerator: str) -> float:
        if accelerator == "gpu":
            return gpu_utilization(self.max_temp, self.telemetry)
        if accelerator == "tpu":
            return tpu_utilization()
        if accelerator == "cpu":
            return cpu_utilization()
        return 0.0

    # --------------------------------------------------------------
    def get_utilization(self) -> Dict[str, float]:
        return {acc: self._utilization(acc) for acc in self.queues}

    # --------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            ran = False
            for acc, queue in self.queues.items():
                if queue and self._utilization(acc) < self.max_util:
                    job = queue.popleft()
                    job()
                    ran = True
            if not ran:
                time.sleep(self.check_interval)


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
        global list_cpus, list_gpus, list_fpgas, list_loihi, list_analog, analog_backend
        if list_cpus is None:
            try:
                from .hardware_detect import (
                    list_cpus,
                    list_gpus,
                    list_fpgas,
                    list_loihi,
                    list_analog,
                )
                from . import hardware_backends as analog_backend
            except Exception:  # pragma: no cover - fallback when deps missing
                def list_cpus() -> list[str]:
                    return ["cpu0"]

                def list_gpus() -> list[str]:
                    return []

                def list_fpgas() -> list[str]:
                    return []

                def list_loihi() -> list[str]:
                    return []

                def list_analog() -> list[str]:
                    return []
                analog_backend = object()
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
        if device == "gpu":
            return gpu_utilization(telemetry=self.telemetry)
        if device == "cpu":
            return cpu_utilization()
        if device == "analog":
            return analog_utilization(analog_backend)
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


class EnergyAwareScheduler(AdaptiveScheduler):
    """Pause or migrate jobs when carbon intensity is high."""

    def __init__(
        self,
        budget: ComputeBudgetTracker,
        run_id: str,
        max_mem: float = 0.9,
        check_interval: float = 1.0,
        window: int = 3,
        min_improvement: float = 0.01,
        intensity_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            budget,
            run_id,
            max_mem=max_mem,
            check_interval=check_interval,
            window=window,
            min_improvement=min_improvement,
        )
        self.intensity_threshold = intensity_threshold
        self.queues = {k: [] for k in self.queues}

    # --------------------------------------------------------------
    def add(
        self,
        job: Callable[[], None],
        *,
        device: str = "gpu",
        region: str | None = None,
    ) -> None:
        cost = self.telemetry.get_carbon_intensity(region)
        self.queues.setdefault(device, []).append((job, cost, region))

    # --------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            if not any(self.queues.values()):
                time.sleep(self.check_interval)
                continue
            if self._should_pause():
                time.sleep(self.check_interval)
                continue
            candidates: list[tuple[float, str, int, Callable[[], None], str | None]] = []
            for dev, queue in self.queues.items():
                if queue and self._device_utilization(dev) < self.max_mem:
                    for i, (job, cost, region) in enumerate(queue):
                        candidates.append((cost, dev, i, job, region))
            if candidates:
                _, dev, idx, job, region = min(candidates, key=lambda x: x[0])
                self.queues[dev].pop(idx)
                intensity = self.telemetry.get_carbon_intensity(region)
                if intensity > self.intensity_threshold:
                    data = self.telemetry.carbon_data or {"default": intensity}
                    best = min(data, key=lambda r: self.telemetry.get_carbon_intensity(r))
                    best_cost = self.telemetry.get_carbon_intensity(best)
                    self.queues[dev].append((job, best_cost, best))
                    time.sleep(self.check_interval)
                    continue
                res = job()
                if isinstance(res, (int, float)):
                    self.record_improvement(float(res))
            else:
                time.sleep(self.check_interval)
        self.budget.stop()


class BatteryAwareScheduler(AdaptiveScheduler):
    """Pause jobs when system battery level is low."""

    def __init__(
        self,
        budget: ComputeBudgetTracker,
        run_id: str,
        max_mem: float = 0.9,
        check_interval: float = 1.0,
        window: int = 3,
        min_improvement: float = 0.01,
        battery_threshold: float = 0.2,
    ) -> None:
        super().__init__(
            budget,
            run_id,
            max_mem=max_mem,
            check_interval=check_interval,
            window=window,
            min_improvement=min_improvement,
        )
        self.battery_threshold = battery_threshold

    # --------------------------------------------------------------
    def _battery_level(self) -> float:
        lvl = battery_level()
        self.telemetry.metrics["battery"] = lvl * 100
        return lvl

    # --------------------------------------------------------------
    def _should_pause(self) -> bool:  # type: ignore[override]
        if self._battery_level() < self.battery_threshold:
            return True
        return super()._should_pause()


__all__ = [
    "BudgetAwareScheduler",
    "AcceleratorScheduler",
    "AdaptiveScheduler",
    "EnergyAwareScheduler",
    "BatteryAwareScheduler",
]
