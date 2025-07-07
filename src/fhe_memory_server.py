from __future__ import annotations

from concurrent import futures
from typing import Any, Iterable

import numpy as np

try:
    import grpc  # type: ignore
    from . import memory_pb2, fhe_memory_pb2, fhe_memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False

try:
    import tenseal as ts  # type: ignore
    _HAS_TENSEAL = True
except Exception:  # pragma: no cover - optional dependency
    ts = None
    _HAS_TENSEAL = False

from .vector_store import VectorStore


if _HAS_GRPC and _HAS_TENSEAL:

    class FHEMemoryServer(fhe_memory_pb2_grpc.FHEMemoryServiceServicer):
        """gRPC server operating on encrypted vectors via TenSEAL."""

        def __init__(
            self,
            store: VectorStore,
            ctx: "ts.Context",
            address: str = "localhost:50900",
            max_workers: int = 4,
        ) -> None:
            self.store = store
            self.ctx = ctx
            self.address = address
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
            fhe_memory_pb2_grpc.add_FHEMemoryServiceServicer_to_server(self, self.server)
            self.server.add_insecure_port(address)

        # --------------------------------------------------
        def Push(self, request: fhe_memory_pb2.FHEPushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            enc = ts.CKKSVector.load(self.ctx, request.vector)
            vec = np.array(enc.decrypt(), dtype=np.float32)
            meta = request.metadata if request.metadata else None
            self.store.add(vec.reshape(1, -1), metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def Query(self, request: fhe_memory_pb2.FHEQueryRequest, context) -> fhe_memory_pb2.FHEQueryReply:  # noqa: N802
            enc_q = ts.CKKSVector.load(self.ctx, request.vector)
            q = np.array(enc_q.decrypt(), dtype=np.float32)
            out, meta = self.store.search(q, k=int(request.k))
            enc_out = ts.ckks_vector(self.ctx, out.reshape(-1).tolist())
            return fhe_memory_pb2.FHEQueryReply(
                vectors=enc_out.serialize(),
                metadata=["" if m is None else str(m) for m in meta],
            )

        def start(self) -> None:
            self.server.start()

        def stop(self, grace: float = 0) -> None:
            self.server.stop(grace)


    class FHEMemoryClient:
        """Thin client for :class:`FHEMemoryServer`."""

        def __init__(self, address: str, ctx: "ts.Context") -> None:
            self.ctx = ctx
            self.channel = grpc.insecure_channel(address)
            self.stub = fhe_memory_pb2_grpc.FHEMemoryServiceStub(self.channel)

        def add(self, vector: np.ndarray, metadata: Any | None = None) -> None:
            enc = ts.ckks_vector(self.ctx, np.asarray(vector, dtype=np.float32).ravel().tolist())
            req = fhe_memory_pb2.FHEPushRequest(
                vector=enc.serialize(),
                metadata="" if metadata is None else str(metadata),
            )
            self.stub.Push(req)

        def search(self, vector: np.ndarray, k: int = 5):
            enc_q = ts.ckks_vector(self.ctx, np.asarray(vector, dtype=np.float32).ravel().tolist())
            req = fhe_memory_pb2.FHEQueryRequest(vector=enc_q.serialize(), k=k)
            reply = self.stub.Query(req)
            enc_out = ts.CKKSVector.load(self.ctx, reply.vectors)
            dim = vector.size
            out = np.array(enc_out.decrypt(), dtype=np.float32).reshape(-1, dim)
            return out, list(reply.metadata)

        def close(self) -> None:
            self.channel.close()


__all__ = ["FHEMemoryServer", "FHEMemoryClient"]
