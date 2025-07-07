"""Fine-grained profiling context manager."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

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

    torch = type("torch", (), {"cuda": _DummyCuda})()  # type: ignore

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints
    from .telemetry import TelemetryLogger


class FineGrainedProfiler:
    """Context manager recording CPU time and GPU memory usage."""

    def __init__(
        self,
        callback: Optional[Callable[[float, float], None]] = None,
        *,
        logger: Optional[TelemetryLogger] = None,
        buffer: Optional[List[Dict[str, float]]] = None,
        label: str | None = None,
    ) -> None:
        self.callback = callback
        self.logger = logger
        self.buffer = buffer
        self.label = label

    def __enter__(self) -> "FineGrainedProfiler":
        self.cpu_start = time.perf_counter()
        self.gpu_start = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        cpu_time = time.perf_counter() - self.cpu_start
        gpu_mem = (
            torch.cuda.memory_allocated() - self.gpu_start
            if torch.cuda.is_available()
            else 0
        )
        stats = {"cpu_time": cpu_time, "gpu_mem": gpu_mem}
        if self.label:
            stats["label"] = self.label
        if self.buffer is not None:
            self.buffer.append(stats)
        if self.logger is not None:
            self.logger.register_profiler(stats)
        if self.callback:
            try:
                self.callback(cpu_time, gpu_mem)
            except Exception:
                pass


__all__ = ["FineGrainedProfiler"]
