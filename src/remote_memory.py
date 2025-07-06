import torch
from typing import Iterable, Tuple, List, Any

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


class RemoteMemory:
    """Thin gRPC client for :class:`~asi.hierarchical_memory.MemoryServer`."""

    def __init__(self, address: str) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for RemoteMemory")
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Send vectors and optional metadata to the remote store."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        self.add_batch(x, metadata)

    def add_batch(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Send a batch of vectors to the remote store in one RPC."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        metas = list(metadata) if metadata is not None else [None] * len(x)
        items = [
            memory_pb2.PushRequest(
                vector=vec.detach().cpu().view(-1).tolist(),
                metadata="" if meta is None else str(meta),
            )
            for vec, meta in zip(x, metas)
        ]
        req = memory_pb2.PushBatchRequest(items=items)
        self.stub.PushBatch(req)

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[str]]:
        """Query nearest vectors from the remote store."""
        if query.ndim == 1:
            q_batch = query.unsqueeze(0)
            vecs, metas = self.search_batch(q_batch, k)
            return vecs[0], metas[0]
        else:
            vecs, metas = self.search_batch(query, k)
            return vecs, metas

    def search_batch(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[List[str]]]:
        """Query multiple vectors from the remote store."""
        if query.ndim == 1:
            query = query.unsqueeze(0)
        items = [
            memory_pb2.QueryRequest(vector=q.detach().cpu().view(-1).tolist(), k=k)
            for q in query
        ]
        req = memory_pb2.QueryBatchRequest(items=items)
        reply = self.stub.QueryBatch(req)
        dim = query.size(-1)
        out_vecs = []
        out_meta = []
        for r in reply.items:
            out_vecs.append(torch.tensor(r.vectors).reshape(-1, dim))
            out_meta.append(list(r.metadata))
        return torch.stack(out_vecs), out_meta

    def delete(self, *, tag: Any) -> None:
        """Delete by tag -- not implemented without server support."""
        raise NotImplementedError("Delete RPC not implemented")


__all__ = ["RemoteMemory"]
