try:
    import torch
except Exception:  # pragma: no cover - allow running without torch
    from .hierarchical_memory import torch
from typing import Iterable, Tuple, List, Any

from . import memory_pb2, memory_pb2_grpc
import numpy as np

class MemoryClientBase:
    """Mixin with batched add and query helpers for gRPC memory clients."""

    stub: memory_pb2_grpc.MemoryServiceStub

    def add_batch(self, vectors: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Send a batch of vectors to the remote store."""
        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)
        metas = list(metadata) if metadata is not None else [None] * len(vectors)
        items = [
            memory_pb2.PushRequest(
                vector=v.detach().cpu().numpy().reshape(-1).tolist(),
                metadata="" if m is None else str(m),
            )
            for v, m in zip(vectors, metas)
        ]
        req = memory_pb2.PushBatchRequest(items=items)
        self.stub.PushBatch(req)

    def query_batch(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[List[str]]]:
        """Query multiple vectors from the remote store."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        items = [
            memory_pb2.QueryRequest(
                vector=q.detach().cpu().numpy().reshape(-1).tolist(), k=k
            )
            for q in query
        ]
        req = memory_pb2.QueryBatchRequest(items=items)
        reply = self.stub.QueryBatch(req)
        dim = query.size(-1)
        out_vecs = []
        out_meta = []
        for r in reply.items:
            arr = np.asarray(r.vectors, dtype=np.float32).reshape(-1, dim)
            out_vecs.append(torch.from_numpy(arr))
            out_meta.append(list(r.metadata))
        return torch.stack(out_vecs), out_meta

__all__ = ["MemoryClientBase"]
