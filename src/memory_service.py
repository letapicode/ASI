"""Helper to expose :class:`HierarchicalMemory` over gRPC."""

from __future__ import annotations

from .hierarchical_memory import HierarchicalMemory, MemoryServer
from .telemetry import TelemetryLogger
from .zero_trust_memory_server import ZeroTrustMemoryServer
from .blockchain_provenance_ledger import BlockchainProvenanceLedger

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
    ledger: BlockchainProvenanceLedger | None = None,
) -> MemoryServer:
    """Start a ``MemoryServer`` at ``address`` and return it.

    If ``ledger`` is provided, a :class:`ZeroTrustMemoryServer` is started
    which validates access tokens against the ledger.
    """
    server: MemoryServer
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
    elif ledger is not None:
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
