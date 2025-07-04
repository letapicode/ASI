"""Helper to expose :class:`HierarchicalMemory` over gRPC."""

from __future__ import annotations

from .hierarchical_memory import HierarchicalMemory, MemoryServer
from .telemetry import TelemetryLogger


def serve(
    memory: HierarchicalMemory,
    address: str,
    max_workers: int = 4,
    telemetry: TelemetryLogger | None = None,
) -> MemoryServer:
    """Start a :class:`MemoryServer` at ``address`` and return it."""
    server = MemoryServer(
        memory, address=address, max_workers=max_workers, telemetry=telemetry
    )
    server.start()
    return server


__all__ = ["serve", "MemoryServer"]
