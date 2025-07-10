from __future__ import annotations

from .hierarchical_memory import HierarchicalMemory

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


if _HAS_GRPC:

    class QuantizedMemoryServer(HierarchicalMemory.MemoryServer):
        """MemoryServer variant for quantized search."""

        def __init__(
            self,
            memory: HierarchicalMemory,
            address: str = "localhost:50120",
            max_workers: int = 4,
            telemetry: "TelemetryLogger | None" = None,
        ) -> None:
            super().__init__(memory, address, max_workers, telemetry)


__all__ = ["QuantizedMemoryServer"]
