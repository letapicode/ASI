"""Unified gRPC memory clients.

This module exposes ``MemoryClientBase`` providing ``add_batch`` and
``query_batch`` helpers plus concrete clients:
``RemoteMemoryClient``, ``QuantumMemoryClient``, ``QuantizedMemoryClient`` and
``EdgeMemoryClient``.
"""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Iterable, Tuple, List, Any, Deque

try:
    import torch
except Exception:  # pragma: no cover - allow running without torch
    from .hierarchical_memory import torch

import numpy as np

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False
    memory_pb2 = memory_pb2_grpc = None  # type: ignore


class MemoryClientBase:
    """Mixin with batched add and query helpers for gRPC memory clients."""

    stub: memory_pb2_grpc.MemoryServiceStub  # type: ignore

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


class RemoteMemoryClient(MemoryClientBase):
    """Thin gRPC client for ``MemoryServer``."""

    def __init__(self, address: str) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for RemoteMemoryClient")
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.add_batch(x, metadata)

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[str]]:
        if query.dim() == 1:
            vecs, metas = self.query_batch(query.unsqueeze(0), k)
            return vecs[0], metas[0]
        vecs, metas = self.query_batch(query, k)
        return vecs, metas

    def delete(self, *, tag: Any) -> None:
        raise NotImplementedError("Delete RPC not implemented")


class QuantumMemoryClient:
    """Thin client for ``QuantumMemoryServer``."""

    def __init__(self, address: str) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for QuantumMemoryClient")
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)

    def add(self, vector: np.ndarray, metadata: Any | None = None) -> None:
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


class QuantizedMemoryClient(MemoryClientBase):
    """Thin client for ``QuantizedMemoryServer``."""

    def __init__(self, address: str) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for QuantizedMemoryClient")
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)


class EdgeMemoryClient:
    """Buffering client that streams vectors to ``RemoteMemoryClient``."""

    def __init__(self, address: str, buffer_size: int = 32, sync_interval: float = 2.0) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for EdgeMemoryClient")
        self.remote = RemoteMemoryClient(address)
        self.buffer_size = buffer_size
        self.sync_interval = sync_interval
        self._vec_buf: List[torch.Tensor] = []
        self._meta_buf: List[Any] = []
        self._queue: Deque[tuple[str, torch.Tensor | None, Any | None]] = deque()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()

    def _sync_loop(self) -> None:
        while not self._stop.is_set():
            if self._queue:
                try:
                    self._flush_queue()
                except Exception:  # pragma: no cover - network failure
                    pass
            time.sleep(self.sync_interval)

    def _flush_queue(self) -> None:
        ops = list(self._queue)
        self._queue.clear()
        for op, vec, meta in ops:
            if op == "add" and vec is not None:
                self.remote.add(vec.unsqueeze(0), [meta])
            elif op == "delete":
                try:
                    self.remote.delete(tag=meta)
                except Exception:
                    self._queue.append((op, vec, meta))

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        metas = list(metadata) if metadata is not None else [None] * len(x)
        for vec, meta in zip(x, metas):
            self._vec_buf.append(vec)
            self._meta_buf.append(meta)
            if len(self._vec_buf) >= self.buffer_size:
                self.flush()

    def flush(self) -> None:
        if not self._vec_buf:
            return
        batch = torch.stack(self._vec_buf)
        try:
            self.remote.add(batch, self._meta_buf)
        except Exception:  # pragma: no cover - network failure
            for vec, meta in zip(self._vec_buf, self._meta_buf):
                self._queue.append(("add", vec, meta))
        else:
            self._vec_buf.clear()
            self._meta_buf.clear()

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[str]]:
        self.flush()
        return self.remote.search(query, k)

    def delete(self, *, tag: Any) -> None:
        try:
            self.remote.delete(tag=tag)
        except Exception:  # pragma: no cover - network failure
            self._queue.append(("delete", None, tag))

    def close(self) -> None:
        self.flush()
        self._stop.set()
        self._thread.join(timeout=0.1)
        if self._queue:
            try:
                self._flush_queue()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> "EdgeMemoryClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def push_remote(
    address: str,
    vector: torch.Tensor,
    metadata: Any | None = None,
    timeout: float = 5.0,
) -> bool:
    """Send ``vector`` to a remote ``MemoryServer``."""
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
    """Query vectors from a remote ``MemoryServer``."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    with grpc.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        req = memory_pb2.QueryRequest(vector=vector.detach().cpu().view(-1).tolist(), k=k)
        reply = stub.Query(req, timeout=timeout)
        vec = torch.tensor(reply.vectors).reshape(-1, vector.size(-1))
        return vec, list(reply.metadata)


def push_batch_remote(
    address: str,
    vectors: torch.Tensor,
    metadata: Iterable[Any] | None = None,
    timeout: float = 5.0,
) -> bool:
    """Send multiple ``vectors`` to a remote ``MemoryServer``."""
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
    """Query ``vectors`` from a remote ``MemoryServer`` in batch."""
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
    """Asynchronously send ``vector`` to a remote ``MemoryServer``."""
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
    """Asynchronously query vectors from a remote ``MemoryServer``."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    async with grpc.aio.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        req = memory_pb2.QueryRequest(vector=vector.detach().cpu().view(-1).tolist(), k=k)
        reply = await stub.Query(req, timeout=timeout)
        vec = torch.tensor(reply.vectors).reshape(-1, vector.size(-1))
        return vec, list(reply.metadata)


async def push_batch_remote_async(
    address: str,
    vectors: torch.Tensor,
    metadata: Iterable[Any] | None = None,
    timeout: float = 5.0,
) -> bool:
    """Asynchronously send multiple ``vectors`` to a remote ``MemoryServer``."""
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
    """Asynchronously query ``vectors`` from a remote ``MemoryServer`` in batch."""
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
    "MemoryClientBase",
    "RemoteMemoryClient",
    "QuantumMemoryClient",
    "QuantizedMemoryClient",
    "EdgeMemoryClient",
    "push_remote",
    "query_remote",
    "push_remote_async",
    "query_remote_async",
    "push_batch_remote",
    "query_batch_remote",
    "push_batch_remote_async",
    "query_batch_remote_async",
]
