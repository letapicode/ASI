from __future__ import annotations

import threading
import time
from typing import Callable

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
        def get_device_properties(_: int):
            class _P:
                total_memory = 1
            return _P()

    torch = type("torch", (), {"cuda": _DummyCuda})()  # type: ignore

from .adaptive_scheduler import AdaptiveScheduler


class EnergyAwareScheduler(AdaptiveScheduler):
    """Pause or migrate jobs when carbon intensity is high."""

    def __init__(
        self,
        budget,
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


__all__ = ["EnergyAwareScheduler"]

