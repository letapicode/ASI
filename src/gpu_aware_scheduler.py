"""Dispatch jobs based on GPU usage."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import torch


class GPUAwareScheduler:
    """Queue jobs and execute when GPU memory is free."""

    def __init__(self, max_mem: float = 0.9, check_interval: float = 1.0) -> None:
        self.max_mem = max_mem
        self.check_interval = check_interval
        self.queue: deque[Callable[[], None]] = deque()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def add(self, job: Callable[[], None]) -> None:
        self.queue.append(job)

    def _loop(self) -> None:
        while True:
            if not self.queue:
                time.sleep(self.check_interval)
                continue
            mem = (
                torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else 0.0
            )
            if mem < self.max_mem:
                job = self.queue.popleft()
                job()
            else:
                time.sleep(self.check_interval)


__all__ = ["GPUAwareScheduler"]
