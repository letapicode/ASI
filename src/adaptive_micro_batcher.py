"""Adaptive micro-batching based on GPU memory usage."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Tuple, Any

import torch

from .telemetry import TelemetryLogger


class AdaptiveMicroBatcher:
    """Dynamically tune micro-batch size using memory statistics."""

    def __init__(
        self,
        batch_size: int,
        min_size: int = 1,
        max_size: int | None = None,
        high_mem: float = 0.9,
        low_mem: float = 0.5,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.min_size = min_size
        self.max_size = max_size or batch_size
        self.high_mem = high_mem
        self.low_mem = low_mem
        self.telemetry = telemetry or TelemetryLogger(interval=0.5)

    # --------------------------------------------------------------
    def start(self) -> None:
        self.telemetry.start()

    def stop(self) -> None:
        self.telemetry.stop()

    # --------------------------------------------------------------
    def _gpu_mem(self) -> float:
        stats = self.telemetry.get_stats()
        mem = stats.get("gpu_mem")
        if mem is None and torch.cuda.is_available():
            mem = (
                torch.cuda.memory_allocated()
                / torch.cuda.get_device_properties(0).total_memory
                * 100.0
            )
        return float(mem or 0.0)

    # --------------------------------------------------------------
    def tick(self) -> int:
        """Update ``batch_size`` based on current memory usage."""
        mem = self._gpu_mem() / 100.0
        if mem > self.high_mem and self.batch_size > self.min_size:
            self.batch_size = max(self.min_size, self.batch_size // 2)
        elif mem < self.low_mem and self.batch_size < self.max_size:
            self.batch_size = min(self.max_size, self.batch_size * 2)
        return self.batch_size

    # --------------------------------------------------------------
    def micro_batches(self, data: Iterable[Any]) -> Iterator[List[Any]]:
        """Yield data in micro-batches with adaptive sizing."""
        buf: List[Any] = []
        for item in data:
            buf.append(item)
            if len(buf) >= self.batch_size:
                yield buf
                buf = []
                self.tick()
        if buf:
            yield buf


__all__ = ["AdaptiveMicroBatcher"]

