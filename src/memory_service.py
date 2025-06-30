"""Helper to expose :class:`HierarchicalMemory` over gRPC."""

from __future__ import annotations

from .hierarchical_memory import HierarchicalMemory, MemoryServer


def serve(memory: HierarchicalMemory, address: str, max_workers: int = 4) -> MemoryServer:
    """Start a :class:`MemoryServer` at ``address`` and return it."""
    server = MemoryServer(memory, address=address, max_workers=max_workers)
    server.start()
    return server


__all__ = ["serve", "MemoryServer"]
