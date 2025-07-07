"""Helper to expose :class:`HierarchicalMemory` over gRPC."""

from __future__ import annotations

from .hierarchical_memory import HierarchicalMemory, MemoryServer
from .telemetry import TelemetryLogger
from .zero_trust_memory_server import ZeroTrustMemoryServer
from .blockchain_provenance_ledger import BlockchainProvenanceLedger


def serve(
    memory: HierarchicalMemory,
    address: str,
    max_workers: int = 4,
    telemetry: TelemetryLogger | None = None,
    ledger: BlockchainProvenanceLedger | None = None,
) -> MemoryServer:
    """Start a ``MemoryServer`` at ``address`` and return it.

    If ``ledger`` is provided, a :class:`ZeroTrustMemoryServer` is started
    which validates access tokens against the ledger.
    """
    server: MemoryServer
    if ledger is not None:
        if ZeroTrustMemoryServer is None:
            raise ImportError("grpcio is required for ZeroTrustMemoryServer")
        server = ZeroTrustMemoryServer(
            memory,
            ledger,
            address=address,
            max_workers=max_workers,
            telemetry=telemetry,
        )
    else:
        server = MemoryServer(
            memory, address=address, max_workers=max_workers, telemetry=telemetry
        )
    server.start()
    return server


__all__ = ["serve", "MemoryServer", "ZeroTrustMemoryServer"]
