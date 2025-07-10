from __future__ import annotations

import torch
from typing import Iterable, Tuple, List, Any

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


class QuantizedMemoryClient:
    """Thin client for :class:`QuantizedMemoryServer`."""

    def __init__(self, address: str) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for QuantizedMemoryClient")
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)

    def add_batch(self, vectors: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        metas = list(metadata) if metadata is not None else [None] * len(vectors)
        items = [
            memory_pb2.PushRequest(vector=v.detach().cpu().view(-1).tolist(), metadata="" if m is None else str(m))
            for v, m in zip(vectors, metas)
        ]
        req = memory_pb2.PushBatchRequest(items=items)
        self.stub.PushBatch(req)

    def query_batch(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[List[str]]]:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        items = [memory_pb2.QueryRequest(vector=q.detach().cpu().view(-1).tolist(), k=k) for q in query]
        req = memory_pb2.QueryBatchRequest(items=items)
        reply = self.stub.QueryBatch(req)
        dim = query.size(-1)
        outs = []
        metas = []
        for r in reply.items:
            outs.append(torch.tensor(r.vectors).reshape(-1, dim))
            metas.append(list(r.metadata))
        return torch.stack(outs), metas


__all__ = ["QuantizedMemoryClient"]
