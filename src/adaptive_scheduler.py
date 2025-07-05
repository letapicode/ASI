from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Any

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
        self.budget = budget
        self.run_id = run_id
        self.telemetry: TelemetryLogger = budget.telemetry
        self.history: Deque[float] = deque(maxlen=window)
        self.min_improvement = min_improvement
        self.max_mem = max_mem
        self.check_interval = check_interval
        self.queue: list[tuple[Callable[[], None], str | None]] = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.budget.start(run_id)
        self.thread.start()

    # --------------------------------------------------------------
    def add(self, job: Callable[[], None], region: str | None = None) -> None:
        """Queue a job with an optional region hint."""
        self.queue.append((job, region))

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
    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self.queue:
                time.sleep(self.check_interval)
                continue
            if self._should_pause():
                time.sleep(self.check_interval)
                continue
            mem = (
                torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else 0.0
            )
            if mem < self.max_mem:
                idx = min(
                    range(len(self.queue)),
                    key=lambda i: self.telemetry.get_cost_index(self.queue[i][1]),
                )
                job, _ = self.queue.pop(idx)
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
        return self.telemetry.get_stats()


__all__ = ["AdaptiveScheduler"]
