from __future__ import annotations

from concurrent import futures
from typing import Any

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False

if _HAS_GRPC:
    import torch
    from .telemetry import TelemetryLogger


    from typing import Callable

    class BaseMemoryServer(memory_pb2_grpc.MemoryServiceServicer):
        """Common gRPC server for memory backends."""

        def __init__(
            self,
            backend: Any,
            address: str = "localhost:50051",
            max_workers: int = 4,
            telemetry: "TelemetryLogger | None" = None,
            service_adder: "Callable[[Any, grpc.Server], None] | None" = None,
        ) -> None:
            self.backend = backend
            self.address = address
            self.telemetry = telemetry
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
            if service_adder is None:
                service_adder = memory_pb2_grpc.add_MemoryServiceServicer_to_server
            service_adder(self, self.server)
            self.server.add_insecure_port(address)

        # --------------------------------------------------------------
        def Push(self, request: memory_pb2.PushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            vec = torch.tensor(request.vector).reshape(1, -1)
            meta = request.metadata if request.metadata else None
            self.backend.add(vec, metadata=[meta])
            if self.telemetry:
                stats = self.telemetry.get_stats()
                stats["push"] = stats.get("push", 0) + 1
            return memory_pb2.PushReply(ok=True)

        def Query(self, request: memory_pb2.QueryRequest, context) -> memory_pb2.QueryReply:  # noqa: N802
            q = torch.tensor(request.vector).reshape(1, -1)
            out, meta = self.backend.search(q, k=int(request.k))
            flat = out.detach().cpu().view(-1).tolist()
            meta = [str(m) for m in meta]
            if self.telemetry:
                stats = self.telemetry.get_stats()
                stats["query"] = stats.get("query", 0) + 1
            return memory_pb2.QueryReply(vectors=flat, metadata=meta)

        def PushBatch(self, request: memory_pb2.PushBatchRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            for item in request.items:
                vec = torch.tensor(item.vector).reshape(1, -1)
                meta = item.metadata if item.metadata else None
                self.backend.add(vec, metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def QueryBatch(self, request: memory_pb2.QueryBatchRequest, context) -> memory_pb2.QueryBatchReply:  # noqa: N802
            replies = []
            for qreq in request.items:
                qt = torch.tensor(qreq.vector).reshape(1, -1)
                out, meta = self.backend.search(qt, k=int(qreq.k))
                flat = out.detach().cpu().view(-1).tolist()
                meta = [str(m) for m in meta]
                replies.append(memory_pb2.QueryReply(vectors=flat, metadata=meta))
            return memory_pb2.QueryBatchReply(items=replies)

        # --------------------------------------------------------------
        def start(self) -> None:
            if self.telemetry:
                self.telemetry.start()
            self.server.start()

        def stop(self, grace: float = 0) -> None:
            self.server.stop(grace)
            if self.telemetry:
                self.telemetry.stop()


__all__ = ["BaseMemoryServer"]
