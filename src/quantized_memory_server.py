from __future__ import annotations

from .hierarchical_memory import HierarchicalMemory
from .base_memory_server import BaseMemoryServer

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


if _HAS_GRPC:

    class QuantizedMemoryServer(BaseMemoryServer):
        """MemoryServer variant for quantized search."""

        def __init__(
            self,
            memory: HierarchicalMemory,
            address: str = "localhost:50120",
            max_workers: int = 4,
            telemetry: "TelemetryLogger | None" = None,
        ) -> None:
            self.memory = memory
            super().__init__(memory, address=address, max_workers=max_workers, telemetry=telemetry)


__all__ = ["QuantizedMemoryServer"]
