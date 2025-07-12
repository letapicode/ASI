from __future__ import annotations

from typing import Any

from .blockchain_provenance_ledger import BlockchainProvenanceLedger
from .base_memory_server import BaseMemoryServer

try:
    import grpc  # type: ignore
    from . import memory_pb2
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


if _HAS_GRPC:
    import torch
    from . import memory_pb2_grpc

    class ZeroTrustMemoryServer(BaseMemoryServer):
        """Memory server that verifies signed access tokens."""

        def __init__(
            self,
            memory: Any,
            ledger: BlockchainProvenanceLedger,
            address: str = "localhost:50051",
            max_workers: int = 4,
            telemetry: "TelemetryLogger | None" = None,
        ) -> None:
            self.memory = memory
            self.ledger = ledger
            super().__init__(memory, address=address, max_workers=max_workers, telemetry=telemetry)

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
        def Push(
            self, request: memory_pb2.PushRequest, context
        ) -> memory_pb2.PushReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            vec = torch.tensor(request.vector).reshape(1, -1)
            meta = request.metadata if request.metadata else None
            self.memory.add(vec, metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def Query(
            self, request: memory_pb2.QueryRequest, context
        ) -> memory_pb2.QueryReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            q = torch.tensor(request.vector).reshape(1, -1)
            out, meta = self.memory.search(q, k=int(request.k))
            flat = out.detach().cpu().view(-1).tolist()
            meta = [str(m) for m in meta]
            return memory_pb2.QueryReply(vectors=flat, metadata=meta)

        def PushBatch(
            self, request: memory_pb2.PushBatchRequest, context
        ) -> memory_pb2.PushReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            for item in request.items:
                vec = torch.tensor(item.vector).reshape(1, -1)
                meta = item.metadata if item.metadata else None
                self.memory.add(vec, metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def QueryBatch(
            self, request: memory_pb2.QueryBatchRequest, context
        ) -> memory_pb2.QueryBatchReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            replies = []
            for qreq in request.items:
                qt = torch.tensor(qreq.vector).reshape(1, -1)
                out, meta = self.memory.search(qt, k=int(qreq.k))
                flat = out.detach().cpu().view(-1).tolist()
                meta = [str(m) for m in meta]
                replies.append(memory_pb2.QueryReply(vectors=flat, metadata=meta))
            return memory_pb2.QueryBatchReply(items=replies)

        # start/stop inherited from ``BaseMemoryServer``

    __all__ = ["ZeroTrustMemoryServer"]
else:  # pragma: no cover - optional dependency
    ZeroTrustMemoryServer = None  # type: ignore
    __all__ = ["ZeroTrustMemoryServer"]

