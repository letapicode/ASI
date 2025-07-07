"""Helper to expose :class:`HierarchicalMemory` over gRPC."""

from __future__ import annotations

from .hierarchical_memory import HierarchicalMemory, MemoryServer
from .telemetry import TelemetryLogger

try:
    import tenseal as ts  # type: ignore
    _HAS_TENSEAL = True
except Exception:  # pragma: no cover - optional
    ts = None
    _HAS_TENSEAL = False


def serve(
    memory: HierarchicalMemory,
    address: str,
    max_workers: int = 4,
    telemetry: TelemetryLogger | None = None,
    fhe_context: "ts.Context | None" = None,
) -> MemoryServer:
    """Start a :class:`MemoryServer` or ``FHEMemoryServer`` at ``address``."""
    if fhe_context is not None:
        if not _HAS_TENSEAL:
            raise ImportError("tenseal is required for FHEMemoryServer")
        from .fhe_memory_server import FHEMemoryServer  # type: ignore

        server = FHEMemoryServer(
            memory.store if hasattr(memory, "store") else memory,  # type: ignore
            fhe_context,
            address=address,
            max_workers=max_workers,
        )
    else:
        server = MemoryServer(
            memory, address=address, max_workers=max_workers, telemetry=telemetry
        )
    server.start()
    return server


__all__ = ["serve", "MemoryServer"]
