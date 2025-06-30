import numpy as np
import torch
from pathlib import Path
from typing import Iterable, Any, Tuple, List

from concurrent import futures
import grpc

from . import memory_pb2, memory_pb2_grpc

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

    # gRPC server helpers -------------------------------------------------
    def start_server(self, host: str = "0.0.0.0", port: int = 50051) -> grpc.Server:
        """Expose this memory instance over gRPC."""

        class Servicer(memory_pb2_grpc.MemoryServiceServicer):
            def __init__(self, mem: "HierarchicalMemory") -> None:
                self.mem = mem

            def Push(self, request, context):  # noqa: N802
                vec = np.array(request.vectors, dtype=np.float32)
                if vec.size:
                    vec = vec.reshape(-1, request.dim)
                    metas = list(request.metadata) if request.metadata else None
                    self.mem.store.add(vec, metas)
                return memory_pb2.PushReply(ok=True)

            def Query(self, request, context):  # noqa: N802
                q = np.array(request.query, dtype=np.float32)
                if q.size:
                    q = q.reshape(request.dim)
                comps, meta = self.mem.store.search(q, request.k)
                return memory_pb2.QueryReply(
                    vectors=comps.ravel().tolist(),
                    metadata=[str(m) for m in meta],
                    dim=comps.shape[1],
                )

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        memory_pb2_grpc.add_MemoryServiceServicer_to_server(Servicer(self), server)
        server.add_insecure_port(f"{host}:{port}")
        server.start()
        self._grpc_server = server
        return server

    def stop_server(self) -> None:
        """Stop the gRPC server if running."""
        srv = getattr(self, "_grpc_server", None)
        if srv:
            srv.stop(0)

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

    # Remote helpers -------------------------------------------------------
    def push_remote(
        self,
        x: torch.Tensor,
        metadata: Iterable[Any] | None = None,
        address: str = "localhost:50051",
    ) -> None:
        """Send compressed vectors to a remote memory service."""
        self.compressor.add(x)
        comp = self.compressor.encoder(x).detach().cpu().numpy()
        stub = memory_pb2_grpc.MemoryServiceStub(grpc.insecure_channel(address))
        req = memory_pb2.PushRequest(
            vectors=comp.ravel().tolist(),
            metadata=[str(m) for m in metadata] if metadata else [],
            dim=comp.shape[1],
        )
        stub.Push(req)

    def query_remote(
        self,
        query: torch.Tensor,
        k: int = 5,
        address: str = "localhost:50051",
    ) -> Tuple[torch.Tensor, List[str]]:
        """Retrieve vectors from a remote memory service."""
        q = self.compressor.encoder(query).detach().cpu().numpy()
        if q.ndim == 2:
            q = q[0]
        stub = memory_pb2_grpc.MemoryServiceStub(grpc.insecure_channel(address))
        resp = stub.Query(memory_pb2.QueryRequest(query=q.tolist(), k=k, dim=q.shape[0]))
        vecs = np.array(resp.vectors, dtype=np.float32)
        if vecs.size == 0:
            return torch.empty(0, query.size(-1), device=query.device), list(resp.metadata)
        vecs = vecs.reshape(-1, resp.dim)
        comp_t = torch.from_numpy(vecs)
        decoded = self.compressor.decoder(comp_t)
        return decoded.to(query.device), list(resp.metadata)

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
        store_dir = path / "store"
        if store_dir.exists():
            mem.store = FaissVectorStore.load(store_dir)
        else:
            mem.store = VectorStore.load(path / "store.npz")
        return mem



    @classmethod
    async def load_async(cls, path: str | Path, use_async: bool = False) -> "HierarchicalMemory":
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
        store_dir = path / "store"
        if store_dir.exists():
            if use_async:
                mem.store = await AsyncFaissVectorStore.load_async(store_dir)
            else:
                mem.store = FaissVectorStore.load(store_dir)
        else:
            mem.store = VectorStore.load(path / "store.npz")
        return mem
