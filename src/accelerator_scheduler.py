"""Dispatch jobs based on accelerator utilization."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict

import psutil
try:  # pragma: no cover - optional torch dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - allow running without torch
    torch = None  # type: ignore

try:  # pragma: no cover - optional tpu dependency
    import torch_xla.core.xla_model as xm  # type: ignore
    _HAS_XLA = True
except Exception:  # pragma: no cover - allow running without torch_xla
    xm = None  # type: ignore
    _HAS_XLA = False


class AcceleratorScheduler:
    """Queue jobs and execute when device utilization is low."""

    def __init__(self, max_util: float = 0.9, check_interval: float = 1.0) -> None:
        self.max_util = max_util
        self.check_interval = check_interval
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
        if accelerator == "gpu" and torch is not None and torch.cuda.is_available():
            try:
                return (
                    torch.cuda.memory_allocated()
                    / torch.cuda.get_device_properties(0).total_memory
                )
            except Exception:
                return 0.0
        if accelerator == "tpu" and _HAS_XLA:
            try:
                info = xm.get_memory_info("xla:0")
                used = info.get("kb_total", 0) - info.get("kb_free", 0)
                total = info.get("kb_total", 1)
                return used / total
            except Exception:
                return 0.0
        if accelerator == "cpu":
            return psutil.cpu_percent(interval=None) / 100.0
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


__all__ = ["AcceleratorScheduler"]
