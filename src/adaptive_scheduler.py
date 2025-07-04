from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque

import torch

from .compute_budget_tracker import ComputeBudgetTracker
from .gpu_aware_scheduler import GPUAwareScheduler


class AdaptiveScheduler(GPUAwareScheduler):
    """GPU-aware scheduler with compute budget checks and progress gating."""

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
        super().__init__(max_mem=max_mem, check_interval=check_interval)

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
                job = self.queue.popleft()
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


__all__ = ["AdaptiveScheduler"]
