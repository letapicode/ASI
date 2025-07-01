import numpy as np
import torch
from pathlib import Path
from typing import Iterable, Any, Tuple, List

try:
    import grpc  # type: ignore
    from concurrent import futures
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False

from .streaming_compression import StreamingCompressor
from .vector_store import VectorStore, FaissVectorStore
from .async_vector_store import AsyncFaissVectorStore


class HierarchicalMemory:
    """Combine streaming compression with a vector store."""

    def __init__(
        self,
        dim: int,
        compressed_dim: int,
        capacity: int,
        db_path: str | Path | None = None,
        use_async: bool = False,
    ) -> None:
        self.compressor = StreamingCompressor(dim, compressed_dim, capacity)
        self.use_async = use_async
        self._next_id = 0
        if use_async:
            self.store = AsyncFaissVectorStore(dim=compressed_dim, path=db_path)
        else:
            if db_path is None:
                self.store = VectorStore(dim=compressed_dim)
            else:
                self.store = FaissVectorStore(dim=compressed_dim, path=db_path)

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return len(self.store)

    # ------------------------------------------------------------------
    # Multimodal helpers

    def add_modalities(
        self,
        text: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        metadata: Iterable[Any] | None = None,
    ) -> None:
        """Add text/image/audio embeddings with modality metadata."""
        n = None
        for t in (text, images, audio):
            if t is not None:
                n = t.shape[0]
                break
        if n is None:
            return
        if metadata is None:
            metas = [self._next_id + i for i in range(n)]
            self._next_id += n
        else:
            metas = list(metadata)
        if len(metas) != n:
            raise ValueError("metadata length mismatch")
        if text is not None:
            self.add(text, [{"id": m, "modality": "text"} for m in metas])
        if images is not None:
            self.add(images, [{"id": m, "modality": "image"} for m in metas])
        if audio is not None:
            self.add(audio, [{"id": m, "modality": "audio"} for m in metas])

    def add_from_fusion(
        self,
        encoded: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        metadata: Iterable[Any] | None = None,
    ) -> None:
        """Add embeddings returned by ``encode_all``."""
        text, images, audio = encoded
        self.add_modalities(text, images, audio, metadata)

    def search_by_modality(
        self, query: torch.Tensor, k: int = 5, modality: str | None = None
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Retrieve vectors filtered by modality."""
        vecs, metas = self.search(query, k=max(k, len(self)))
        if modality is None:
            return vecs[:k], metas[:k]
        out_vecs = []
        out_meta = []
        for v, m in zip(vecs, metas):
            if isinstance(m, dict) and m.get("modality") == modality:
                out_vecs.append(v)
                out_meta.append(m)
                if len(out_vecs) >= k:
                    break
        if not out_vecs:
            empty = torch.empty(0, query.size(-1), device=query.device)
            return empty, []
        return torch.stack(out_vecs), out_meta

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Compress and store embeddings with optional metadata."""
        if isinstance(self.store, AsyncFaissVectorStore):
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.aadd(x, metadata))
            else:
                return loop.create_task(self.aadd(x, metadata))
        else:
            self.compressor.add(x)
            comp = self.compressor.encoder(x).detach().cpu().numpy()
            self.store.add(comp, metadata)

    def add_multimodal(
        self,
        text: torch.Tensor,
        images: torch.Tensor,
        audio: torch.Tensor,
        metadata: Iterable[Any] | None = None,
    ) -> None:
        """Store averaged multimodal embeddings."""
        vecs = (text + images + audio) / 3.0
        self.add(vecs, metadata)

    def delete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
        """Remove vectors from the store by index or metadata tag."""
        if isinstance(self.store, AsyncFaissVectorStore):
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.adelete(index, tag))
            else:
                return loop.create_task(self.adelete(index, tag))
        else:
            self.store.delete(index=index, tag=tag)

    async def aadd(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Asynchronously compress and store embeddings."""
        self.compressor.add(x)
        comp = self.compressor.encoder(x).detach().cpu().numpy()
        if isinstance(self.store, AsyncFaissVectorStore):
            await self.store.aadd(comp, metadata)
        else:
            self.store.add(comp, metadata)

    async def aadd_multimodal(
        self,
        text: torch.Tensor,
        images: torch.Tensor,
        audio: torch.Tensor,
        metadata: Iterable[Any] | None = None,
    ) -> None:
        """Asynchronously store averaged multimodal embeddings."""
        vecs = (text + images + audio) / 3.0
        await self.aadd(vecs, metadata)

    async def adelete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
        """Asynchronously delete vectors from the store."""
        import asyncio

        if isinstance(self.store, AsyncFaissVectorStore):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.store.delete, index, tag)
        else:
            self.store.delete(index=index, tag=tag)

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[Any]]:
        """Retrieve top-k decoded vectors and their metadata."""
        if isinstance(self.store, AsyncFaissVectorStore):
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.asearch(query, k))
            else:
                return loop.create_task(self.asearch(query, k))
        q = self.compressor.encoder(query).detach().cpu().numpy()
        if q.ndim == 2:
            q = q[0]
        comp_vecs, meta = self.store.search(q, k)
        if comp_vecs.shape[0] == 0:
            empty = torch.empty(0, query.size(-1), device=query.device)
            return empty, meta
        comp_t = torch.from_numpy(comp_vecs)
        decoded = self.compressor.decoder(comp_t)
        return decoded.to(query.device), meta

    async def asearch(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[Any]]:
        """Asynchronously retrieve vectors and metadata."""
        q = self.compressor.encoder(query).detach().cpu().numpy()
        if q.ndim == 2:
            q = q[0]
        if isinstance(self.store, AsyncFaissVectorStore):
            comp_vecs, meta = await self.store.asearch(q, k)
        else:
            comp_vecs, meta = self.store.search(q, k)
        if comp_vecs.shape[0] == 0:
            empty = torch.empty(0, query.size(-1), device=query.device)
            return empty, meta
        comp_t = torch.from_numpy(comp_vecs)
        decoded = self.compressor.decoder(comp_t)
        return decoded.to(query.device), meta

    def save(self, path: str | Path) -> None:
        """Persist compressor state and vector store to ``path``."""
        if isinstance(self.store, AsyncFaissVectorStore):
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.save_async(path))
            else:
                return loop.create_task(self.save_async(path))
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        comp_state = {
            "dim": self.compressor.encoder.in_features,
            "compressed_dim": self.compressor.encoder.out_features,
            "capacity": self.compressor.buffer.capacity,
            "buffer": [t.cpu() for t in self.compressor.buffer.data],
            "count": self.compressor.buffer.count,
            "encoder": self.compressor.encoder.state_dict(),
            "decoder": self.compressor.decoder.state_dict(),
            "next_id": self._next_id,
        }
        torch.save(comp_state, path / "compressor.pt")
        if isinstance(self.store, FaissVectorStore):
            self.store.save(path / "store")
        else:
            self.store.save(path / "store.npz")

    async def save_async(self, path: str | Path) -> None:
        """Asynchronously persist compressor state and vector store."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        comp_state = {
            "dim": self.compressor.encoder.in_features,
            "compressed_dim": self.compressor.encoder.out_features,
            "capacity": self.compressor.buffer.capacity,
            "buffer": [t.cpu() for t in self.compressor.buffer.data],
            "count": self.compressor.buffer.count,
            "encoder": self.compressor.encoder.state_dict(),
            "decoder": self.compressor.decoder.state_dict(),
            "next_id": self._next_id,
        }
        torch.save(comp_state, path / "compressor.pt")
        if isinstance(self.store, AsyncFaissVectorStore):
            await self.store.save_async(path / "store")
        elif isinstance(self.store, FaissVectorStore):
            self.store.save(path / "store")
        else:
            self.store.save(path / "store.npz")

    @classmethod
    def load(cls, path: str | Path, use_async: bool = False) -> "HierarchicalMemory":
        """Load memory from ``save()`` output."""
        if use_async:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(cls.load_async(path, use_async=True))
            else:
                return loop.create_task(cls.load_async(path, use_async=True))
        path = Path(path)
        state = torch.load(path / "compressor.pt", map_location="cpu")
        mem = cls(
            dim=int(state["dim"]),
            compressed_dim=int(state["compressed_dim"]),
            capacity=int(state["capacity"]),
            use_async=use_async,
        )
        mem.compressor.encoder.load_state_dict(state["encoder"])
        mem.compressor.decoder.load_state_dict(state["decoder"])
        mem.compressor.buffer.data = [t.clone() for t in state["buffer"]]
        mem.compressor.buffer.count = int(state["count"])
        mem._next_id = int(state.get("next_id", 0))
        store_dir = path / "store"
        if store_dir.exists():
            mem.store = FaissVectorStore.load(store_dir)
        else:
            mem.store = VectorStore.load(path / "store.npz")
        return mem

    @classmethod
    async def load_async(
        cls, path: str | Path, use_async: bool = False
    ) -> "HierarchicalMemory":
        """Asynchronously load memory from ``save_async()`` output."""
        path = Path(path)
        state = torch.load(path / "compressor.pt", map_location="cpu")
        mem = cls(
            dim=int(state["dim"]),
            compressed_dim=int(state["compressed_dim"]),
            capacity=int(state["capacity"]),
            use_async=use_async,
        )
        mem.compressor.encoder.load_state_dict(state["encoder"])
        mem.compressor.decoder.load_state_dict(state["decoder"])
        mem.compressor.buffer.data = [t.clone() for t in state["buffer"]]
        mem.compressor.buffer.count = int(state["count"])
        mem._next_id = int(state.get("next_id", 0))
        store_dir = path / "store"
        if store_dir.exists():
            if use_async:
                mem.store = await AsyncFaissVectorStore.load_async(store_dir)
            else:
                mem.store = FaissVectorStore.load(store_dir)
        else:
            mem.store = VectorStore.load(path / "store.npz")
        return mem


if _HAS_GRPC:
    class MemoryServer(memory_pb2_grpc.MemoryServiceServicer):
        """gRPC server exposing a ``HierarchicalMemory`` backend."""

        def __init__(self, memory: HierarchicalMemory, address: str = "localhost:50051", max_workers: int = 4) -> None:
            self.memory = memory
            self.address = address
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
            memory_pb2_grpc.add_MemoryServiceServicer_to_server(self, self.server)
            self.server.add_insecure_port(address)

        def Push(self, request: memory_pb2.PushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            vec = torch.tensor(request.vector).reshape(1, -1)
            meta = request.metadata if request.metadata else None
            self.memory.add(vec, metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def Query(self, request: memory_pb2.QueryRequest, context) -> memory_pb2.QueryReply:  # noqa: N802
            q = torch.tensor(request.vector).reshape(1, -1)
            out, meta = self.memory.search(q, k=int(request.k))
            flat = out.detach().cpu().view(-1).tolist()
            meta = [str(m) for m in meta]
            return memory_pb2.QueryReply(vectors=flat, metadata=meta)

        def start(self) -> None:
            """Start serving requests."""
            self.server.start()

        def stop(self, grace: float = 0) -> None:
            """Stop the server."""
            self.server.stop(grace)


def push_remote(address: str, vector: torch.Tensor, metadata: Any | None = None, timeout: float = 5.0) -> bool:
    """Send ``vector`` to a remote :class:`MemoryServer`."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    with grpc.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        req = memory_pb2.PushRequest(vector=vector.detach().cpu().view(-1).tolist(), metadata="" if metadata is None else str(metadata))
        reply = stub.Push(req, timeout=timeout)
        return reply.ok


def query_remote(address: str, vector: torch.Tensor, k: int = 5, timeout: float = 5.0) -> Tuple[torch.Tensor, List[str]]:
    """Query vectors from a remote :class:`MemoryServer`."""
    if not _HAS_GRPC:
        raise ImportError("grpcio is required for remote memory")
    with grpc.insecure_channel(address) as channel:
        stub = memory_pb2_grpc.MemoryServiceStub(channel)
        req = memory_pb2.QueryRequest(vector=vector.detach().cpu().view(-1).tolist(), k=k)
        reply = stub.Query(req, timeout=timeout)
        vec = torch.tensor(reply.vectors).reshape(-1, vector.size(-1))
        return vec, list(reply.metadata)


async def push_remote_async(
    address: str, vector: torch.Tensor, metadata: Any | None = None, timeout: float = 5.0
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
        req = memory_pb2.QueryRequest(vector=vector.detach().cpu().view(-1).tolist(), k=k)
        reply = await stub.Query(req, timeout=timeout)
        vec = torch.tensor(reply.vectors).reshape(-1, vector.size(-1))
        return vec, list(reply.metadata)
