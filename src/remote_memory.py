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
        metas = list(metadata) if metadata is not None else [None] * len(x)
        for vec, meta in zip(x, metas):
            req = memory_pb2.PushRequest(
                vector=vec.detach().cpu().view(-1).tolist(),
                metadata="" if meta is None else str(meta),
            )
            self.stub.Push(req)

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[str]]:
        """Query nearest vectors from the remote store."""
        req = memory_pb2.QueryRequest(vector=query.detach().cpu().view(-1).tolist(), k=k)
        reply = self.stub.Query(req)
        vec = torch.tensor(reply.vectors).reshape(-1, query.size(-1))
        return vec, list(reply.metadata)


__all__ = ["RemoteMemory"]
