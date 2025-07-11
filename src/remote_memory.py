try:
    import torch
except Exception:  # pragma: no cover - allow running without torch
    from .hierarchical_memory import torch
from typing import Iterable, Tuple, List, Any

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    from .memory_client_base import MemoryClientBase
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False
    MemoryClientBase = object  # type: ignore


class RemoteMemory(MemoryClientBase):
    """Thin gRPC client for :class:`~asi.hierarchical_memory.MemoryServer`."""

    def __init__(self, address: str) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for RemoteMemory")
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Send vectors and optional metadata to the remote store."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.add_batch(x, metadata)


    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[str]]:
        """Query nearest vectors from the remote store."""
        if query.dim() == 1:
            q_batch = query.unsqueeze(0)
            vecs, metas = self.query_batch(q_batch, k)
            return vecs[0], metas[0]
        else:
            vecs, metas = self.query_batch(query, k)
            return vecs, metas

    def delete(self, *, tag: Any) -> None:
        """Delete by tag -- not implemented without server support."""
        raise NotImplementedError("Delete RPC not implemented")


__all__ = ["RemoteMemory"]
