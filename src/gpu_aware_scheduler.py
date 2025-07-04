"""Backward-compatible wrapper for AcceleratorScheduler."""

from __future__ import annotations

from .accelerator_scheduler import AcceleratorScheduler


class GPUAwareScheduler(AcceleratorScheduler):
    """Alias of :class:`AcceleratorScheduler` for GPU jobs."""

    def __init__(self, max_mem: float = 0.9, check_interval: float = 1.0) -> None:
        super().__init__(max_util=max_mem, check_interval=check_interval)


__all__ = ["GPUAwareScheduler"]
