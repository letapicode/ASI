from __future__ import annotations

from .hierarchical_memory import HierarchicalMemory, MemoryServer
from .blockchain_provenance_ledger import BlockchainProvenanceLedger

try:
    import grpc  # type: ignore
    from . import memory_pb2
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False

if _HAS_GRPC:

    class ZeroTrustMemoryServer(MemoryServer):
        """Memory server that verifies signed access tokens."""

        def __init__(
            self,
            memory: HierarchicalMemory,
            ledger: BlockchainProvenanceLedger,
            address: str = "localhost:50051",
            max_workers: int = 4,
            telemetry: "TelemetryLogger | None" = None,
        ) -> None:
            super().__init__(memory, address=address, max_workers=max_workers, telemetry=telemetry)
            self.ledger = ledger

        # --------------------------------------------------
        def _check_token(self, context: grpc.ServicerContext) -> bool:
            md = dict(context.invocation_metadata())
            token = md.get("authorization")
            if not token:
                return False
            try:
                return self.ledger.verify([token])
            except Exception:
                return False

        # --------------------------------------------------
        def Push(self, request: memory_pb2.PushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            return super().Push(request, context)

        def Query(self, request: memory_pb2.QueryRequest, context) -> memory_pb2.QueryReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            return super().Query(request, context)

        def PushBatch(self, request: memory_pb2.PushBatchRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            return super().PushBatch(request, context)

        def QueryBatch(self, request: memory_pb2.QueryBatchRequest, context) -> memory_pb2.QueryBatchReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            return super().QueryBatch(request, context)

    __all__ = ["ZeroTrustMemoryServer"]
else:  # pragma: no cover - optional dependency
    ZeroTrustMemoryServer = None  # type: ignore
    __all__ = ["ZeroTrustMemoryServer"]
