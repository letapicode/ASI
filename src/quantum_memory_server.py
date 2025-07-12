from __future__ import annotations

from typing import Iterable, Any

import numpy as np

from .vector_stores import VectorStore
from .base_memory_server import BaseMemoryServer

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


if _HAS_GRPC:

    class QuantumMemoryServer(BaseMemoryServer):
        """gRPC server exposing a :class:`VectorStore` with quantum search."""

        def __init__(
            self,
            store: VectorStore,
            address: str = "localhost:50520",
            max_workers: int = 4,
        ) -> None:
            self.store = store
            super().__init__(store, address=address, max_workers=max_workers)

        # --------------------------------------------------------------
        def Push(self, request: memory_pb2.PushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            vec = np.asarray(request.vector, dtype=np.float32)
            meta = request.metadata if request.metadata else None
            self.store.add(vec.reshape(1, -1), metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def Query(self, request: memory_pb2.QueryRequest, context) -> memory_pb2.QueryReply:  # noqa: N802
            q = np.asarray(request.vector, dtype=np.float32)
            out, meta = self.store.search(q, k=int(request.k), quantum=True)
            return memory_pb2.QueryReply(
                vectors=out.reshape(-1).tolist(),
                metadata=["" if m is None else str(m) for m in meta],
            )

        def PushBatch(self, request: memory_pb2.PushBatchRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            vecs: list[np.ndarray] = []
            metas: list[Any] = []
            for item in request.items:
                vecs.append(np.asarray(item.vector, dtype=np.float32))
                metas.append(item.metadata if item.metadata else None)
            if vecs:
                arr = np.stack(vecs, axis=0)
                self.store.add(arr, metadata=metas)
            return memory_pb2.PushReply(ok=True)

        def QueryBatch(self, request: memory_pb2.QueryBatchRequest, context) -> memory_pb2.QueryBatchReply:  # noqa: N802
            replies = []
            for q in request.items:
                vec = np.asarray(q.vector, dtype=np.float32)
                out, meta = self.store.search(vec, k=int(q.k), quantum=True)
                replies.append(
                    memory_pb2.QueryReply(
                        vectors=out.reshape(-1).tolist(),
                        metadata=["" if m is None else str(m) for m in meta],
                    )
                )
            return memory_pb2.QueryBatchReply(items=replies)

        # start/stop inherited from ``BaseMemoryServer``


__all__ = ["QuantumMemoryServer"]
