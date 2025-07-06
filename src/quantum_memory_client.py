from __future__ import annotations

import numpy as np

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


class QuantumMemoryClient:
    """Thin client for :class:`QuantumMemoryServer`."""

    def __init__(self, address: str) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for QuantumMemoryClient")
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)

    def add(self, vector: np.ndarray, metadata: any | None = None) -> None:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1).tolist()
        meta = "" if metadata is None else str(metadata)
        req = memory_pb2.PushRequest(vector=arr, metadata=meta)
        self.stub.Push(req)

    def search(self, vector: np.ndarray, k: int = 5):
        arr = np.asarray(vector, dtype=np.float32).reshape(-1).tolist()
        req = memory_pb2.QueryRequest(vector=arr, k=k)
        reply = self.stub.Query(req)
        dim = len(arr)
        vecs = np.array(reply.vectors, dtype=np.float32).reshape(-1, dim)
        return vecs, list(reply.metadata)

    def close(self) -> None:
        self.channel.close()


__all__ = ["QuantumMemoryClient"]
