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


def push_remote(
    address: str,
    vector: torch.Tensor,
    metadata: Any | None = None,
    timeout: float = 5.0,
) -> bool:
    """Send ``vector`` to a remote :class:`MemoryServer`."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    with grpc.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        req = memory_pb2.PushRequest(
            vector=vector.detach().cpu().view(-1).tolist(),
            metadata="" if metadata is None else str(metadata),
        )
        reply = stub.Push(req, timeout=timeout)
        return reply.ok


def query_remote(
    address: str, vector: torch.Tensor, k: int = 5, timeout: float = 5.0
) -> Tuple[torch.Tensor, List[str]]:
    """Query vectors from a remote :class:`MemoryServer`."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    with grpc.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        req = memory_pb2.QueryRequest(
            vector=vector.detach().cpu().view(-1).tolist(), k=k
        )
        reply = stub.Query(req, timeout=timeout)
        vec = torch.tensor(reply.vectors).reshape(-1, vector.size(-1))
        return vec, list(reply.metadata)


def push_batch_remote(
    address: str,
    vectors: torch.Tensor,
    metadata: Iterable[Any] | None = None,
    timeout: float = 5.0,
) -> bool:
    """Send multiple ``vectors`` to a remote :class:`MemoryServer`."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
    metas = list(metadata) if metadata is not None else [None] * len(vectors)
    with grpc.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        items = [
            memory_pb2.PushRequest(
                vector=v.detach().cpu().view(-1).tolist(),
                metadata="" if m is None else str(m),
            )
            for v, m in zip(vectors, metas)
        ]
        req = memory_pb2.PushBatchRequest(items=items)
        reply = stub.PushBatch(req, timeout=timeout)
        return reply.ok


def query_batch_remote(
    address: str, vectors: torch.Tensor, k: int = 5, timeout: float = 5.0
) -> Tuple[torch.Tensor, List[List[str]]]:
    """Query ``vectors`` from a remote :class:`MemoryServer` in batch."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
    with grpc.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        items = [
            memory_pb2.QueryRequest(vector=v.detach().cpu().view(-1).tolist(), k=k)
            for v in vectors
        ]
        req = memory_pb2.QueryBatchRequest(items=items)
        reply = stub.QueryBatch(req, timeout=timeout)
        dim = vectors.size(-1)
        outs = []
        metas = []
        for r in reply.items:
            outs.append(torch.tensor(r.vectors).reshape(-1, dim))
            metas.append(list(r.metadata))
        return torch.stack(outs), metas


async def push_remote_async(
    address: str,
    vector: torch.Tensor,
    metadata: Any | None = None,
    timeout: float = 5.0,
) -> bool:
    """Asynchronously send ``vector`` to a remote :class:`MemoryServer`."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    async with grpc.aio.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        req = memory_pb2.PushRequest(
            vector=vector.detach().cpu().view(-1).tolist(),
            metadata="" if metadata is None else str(metadata),
        )
        reply = await stub.Push(req, timeout=timeout)
        return reply.ok


async def query_remote_async(
    address: str, vector: torch.Tensor, k: int = 5, timeout: float = 5.0
) -> Tuple[torch.Tensor, List[str]]:
    """Asynchronously query vectors from a remote :class:`MemoryServer`."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    async with grpc.aio.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        req = memory_pb2.QueryRequest(
            vector=vector.detach().cpu().view(-1).tolist(), k=k
        )
        reply = await stub.Query(req, timeout=timeout)
        vec = torch.tensor(reply.vectors).reshape(-1, vector.size(-1))
        return vec, list(reply.metadata)


async def push_batch_remote_async(
    address: str,
    vectors: torch.Tensor,
    metadata: Iterable[Any] | None = None,
    timeout: float = 5.0,
) -> bool:
    """Asynchronously send multiple ``vectors`` to a remote :class:`MemoryServer`."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
    metas = list(metadata) if metadata is not None else [None] * len(vectors)
    async with grpc.aio.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        items = [
            memory_pb2.PushRequest(
                vector=v.detach().cpu().view(-1).tolist(),
                metadata="" if m is None else str(m),
            )
            for v, m in zip(vectors, metas)
        ]
        req = memory_pb2.PushBatchRequest(items=items)
        reply = await stub.PushBatch(req, timeout=timeout)
        return reply.ok


async def query_batch_remote_async(
    address: str, vectors: torch.Tensor, k: int = 5, timeout: float = 5.0
) -> Tuple[torch.Tensor, List[List[str]]]:
    """Asynchronously query ``vectors`` from a remote :class:`MemoryServer` in batch."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
    async with grpc.aio.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        items = [
            memory_pb2.QueryRequest(vector=v.detach().cpu().view(-1).tolist(), k=k)
            for v in vectors
        ]
        req = memory_pb2.QueryBatchRequest(items=items)
        reply = await stub.QueryBatch(req, timeout=timeout)
        dim = vectors.size(-1)
        outs = []
        metas = []
        for r in reply.items:
            outs.append(torch.tensor(r.vectors).reshape(-1, dim))
            metas.append(list(r.metadata))
        return torch.stack(outs), metas


__all__ = [
    "RemoteMemory",
    "push_remote",
    "query_remote",
    "push_remote_async",
    "query_remote_async",
    "push_batch_remote",
    "query_batch_remote",
    "push_batch_remote_async",
    "query_batch_remote_async",
]
