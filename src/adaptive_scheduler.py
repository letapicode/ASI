from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict

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
            class P:
                total_memory = 1
            return P()

    torch = type("torch", (), {"cuda": _DummyCuda})()  # type: ignore

from .compute_budget_tracker import ComputeBudgetTracker
from .accelerator_scheduler import AcceleratorScheduler


class AdaptiveScheduler(AcceleratorScheduler):
    """Accelerator-aware scheduler with compute budget checks and progress gating."""

    def __init__(
        self,
        budget: ComputeBudgetTracker,
        run_id: str,
        max_mem: float = 0.9,
        check_interval: float = 1.0,
        window: int = 3,
        min_improvement: float = 0.01,
    ) -> None:
        self.budget = budget
        self.run_id = run_id
        self.history: Deque[float] = deque(maxlen=window)
        self.min_improvement = min_improvement
        self._stop = threading.Event()
        self.budget.start(run_id)
        super().__init__(max_util=max_mem, check_interval=check_interval)

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
    def _loop(self) -> None:  # type: ignore[override]
        while not self._stop.is_set():
            if not any(self.queues[acc] for acc in self.queues):
                time.sleep(self.check_interval)
                continue
            if self._should_pause():
                time.sleep(self.check_interval)
                continue
            ran = False
            for acc, queue in self.queues.items():
                if queue and self._utilization(acc) < self.max_util:
                    job = queue.popleft()
                    res = job()
                    if isinstance(res, (int, float)):
                        self.record_improvement(float(res))
                    ran = True
            if not ran:
                time.sleep(self.check_interval)
        self.budget.stop()

    # --------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.budget.stop()

    # --------------------------------------------------------------
    def report_load(self) -> Dict[str, float]:
        """Return current accelerator utilisation."""
        return self.get_utilization()


__all__ = ["AdaptiveScheduler"]
