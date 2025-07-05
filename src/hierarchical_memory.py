import numpy as np
import torch
from pathlib import Path
from typing import Iterable, Any, Tuple, List, Dict

try:
    import grpc  # type: ignore
    from concurrent import futures
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False

from .streaming_compression import StreamingCompressor, TemporalVectorCompressor
from .knowledge_graph_memory import KnowledgeGraphMemory, TimedTriple
from .vector_store import VectorStore, FaissVectorStore, LocalitySensitiveHashIndex
from .pq_vector_store import PQVectorStore
from .async_vector_store import AsyncFaissVectorStore
from .hopfield_memory import HopfieldMemory
from .data_ingest import CrossLingualTranslator
from .retrieval_rl import RetrievalPolicy


class SSDCache:
    """Simple SSD-backed vector cache."""

    def __init__(self, dim: int, path: str | Path, max_size: int = 10000) -> None:
        self.dim = dim
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.file = self.path / "cache.npz"
        if self.file.exists():
            data = np.load(self.file, allow_pickle=True)
            self.vectors = data["vectors"]
            self.meta = data["meta"].tolist()
        else:
            self.vectors = np.empty((0, dim), dtype=np.float32)
        self.meta: List[Any] = []

    def add(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None]
        metas = list(metadata) if metadata is not None else [None] * arr.shape[0]
        self.vectors = np.concatenate([self.vectors, arr], axis=0)[-self.max_size :]
        self.meta = (self.meta + metas)[-self.max_size :]
        self.save()

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        if self.vectors.size == 0:
            return np.empty((0, self.dim), dtype=np.float32), []
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        scores = self.vectors @ q.T
        idx = np.argsort(scores.ravel())[::-1][:k]
        return self.vectors[idx], [self.meta[i] for i in idx]

    def save(self, path: str | Path | None = None) -> None:
        file = Path(path) / "cache.npz" if path is not None else self.file
        np.savez_compressed(file, vectors=self.vectors, meta=np.array(self.meta, dtype=object))

    @classmethod
    def load(cls, path: str | Path) -> "SSDCache":
        path = Path(path)
        data = np.load(path / "cache.npz", allow_pickle=True)
        cache = cls(int(data["vectors"].shape[1]), path)
        cache.vectors = data["vectors"]
        cache.meta = data["meta"].tolist()
        return cache


class HopfieldStore:
    """Wrapper around :class:`HopfieldMemory` with a VectorStore-like API."""

    def __init__(self, dim: int) -> None:
        self.mem = HopfieldMemory(dim)
        self._meta: List[Any] = []

    def __len__(self) -> int:
        return len(self._meta)

    def add(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        self.mem.store(arr)
        metas = list(metadata) if metadata is not None else [None] * arr.shape[0]
        self._meta.extend(metas)

    def delete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
        pass  # deletion not supported

    def search(self, query: np.ndarray, k: int = 1) -> Tuple[np.ndarray, List[Any]]:
        vec = self.mem.retrieve(query, steps=10)
        out = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        metas = self._meta[:1] if self._meta else [None]
        return out, metas



class HierarchicalMemory:
    """Combine streaming compression with a vector store."""

    def __init__(
        self,
        dim: int,
        compressed_dim: int,
        capacity: int,
        db_path: str | Path | None = None,
        use_async: bool = False,
        use_hopfield: bool = False,
        use_lsh: bool = False,
        use_pq: bool = False,
        lsh_planes: int = 16,
        cache_dir: str | Path | None = None,
        cache_size: int = 1000,
        temporal_decay: float | None = None,
        evict_limit: int | None = None,
        adaptive_evict: bool = False,
        evict_check_interval: int = 100,
        use_kg: bool = False,
        translator: "CrossLingualTranslator | None" = None,
        retrieval_policy: "RetrievalPolicy | None" = None,
    ) -> None:
        if temporal_decay is None:
            self.compressor = StreamingCompressor(dim, compressed_dim, capacity)
        else:
            self.compressor = TemporalVectorCompressor(
                dim, compressed_dim, capacity, decay=temporal_decay
            )
        self.use_async = use_async
        self._next_id = 0
        if use_hopfield:
            self.store = HopfieldStore(dim=compressed_dim)
        elif use_async:
            self.store = AsyncFaissVectorStore(dim=compressed_dim, path=db_path)
        elif use_lsh:
            self.store = LocalitySensitiveHashIndex(dim=compressed_dim, num_planes=lsh_planes)
        elif use_pq:
            self.store = PQVectorStore(dim=compressed_dim, path=db_path)
        else:
            if db_path is None:
                self.store = VectorStore(dim=compressed_dim)
            else:
                self.store = FaissVectorStore(dim=compressed_dim, path=db_path)
        self.cache: SSDCache | None = None
        if cache_dir is not None:
            self.cache = SSDCache(dim=dim, path=cache_dir, max_size=cache_size)
        self.evict_limit = evict_limit
        self._usage: Dict[Any, int] = {}
        self.adaptive_evict = adaptive_evict
        self.evict_check_interval = max(1, evict_check_interval)
        self.hit_count = 0
        self.miss_count = 0
        self.query_count = 0
        self.kg: KnowledgeGraphMemory | None = KnowledgeGraphMemory() if use_kg else None
        self.last_trace: dict | None = None
        self.translator = translator
        self.retrieval_policy = retrieval_policy

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
        target_id = None
        for v, m in zip(vecs, metas):
            if isinstance(m, dict) and torch.allclose(v, query, atol=1e-6):
                target_id = m.get("id")
                break
        matches = [
            (v, m)
            for v, m in zip(vecs, metas)
            if isinstance(m, dict) and m.get("modality") == modality and (
                target_id is None or m.get("id") == target_id
            )
        ]
        if matches:
            vecs_f, metas_f = zip(*matches[:k])
            return torch.stack(list(vecs_f)), list(metas_f)
        # fallback to similarity ranking
        scores = (vecs @ query.view(-1)).cpu()
        filtered = [
            (s.item(), v, m)
            for s, v, m in zip(scores, vecs, metas)
            if isinstance(m, dict) and m.get("modality") == modality
        ]
        if not filtered:
            empty = torch.empty(0, query.size(-1), device=query.device)
            return empty, []
        filtered.sort(key=lambda x: x[0], reverse=True)
        out_vecs = [v for _, v, _ in filtered[:k]]
        out_meta = [m for _, _, m in filtered[:k]]
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
            metas = list(metadata) if metadata is not None else []
            if not metas:
                metas = [self._next_id + i for i in range(comp.shape[0])]
                self._next_id += comp.shape[0]
            self.store.add(comp, metas)
            for m in metas:
                self._usage[m] = 0
            if self.kg is not None:
                triples = [t for t in metas if isinstance(t, tuple) and len(t) == 3]
                if triples:
                    self.kg.add_triples(triples)
            self._evict_if_needed()

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

    def _evict_if_needed(self) -> None:
        if self.evict_limit is None:
            return
        while len(self.store) > self.evict_limit:
            if not self._usage:
                break
            victim = min(self._usage.items(), key=lambda x: x[1])[0]
            self.store.delete(tag=victim)
            self._usage.pop(victim, None)

    def _update_eviction(self) -> None:
        """Adjust ``evict_limit`` based on hit rate."""
        total = self.hit_count + self.miss_count
        if total == 0 or self.evict_limit is None:
            return
        rate = self.hit_count / total
        if rate < 0.5 and self.evict_limit > 1:
            self.evict_limit = max(1, int(self.evict_limit * 0.9))
        elif rate > 0.9:
            self.evict_limit = int(self.evict_limit * 1.1) + 1
        self.hit_count = 0
        self.miss_count = 0
        self.query_count = 0
        self._evict_if_needed()

    def get_stats(self) -> Dict[str, float]:
        """Return current hit/miss statistics."""
        total = self.hit_count + self.miss_count
        rate = self.hit_count / total if total else 0.0
        return {
            "queries": float(total),
            "hits": float(self.hit_count),
            "misses": float(self.miss_count),
            "hit_rate": rate,
            "evict_limit": float(self.evict_limit or 0),
        }

    def query_triples(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[TimedTriple]:
        """Proxy to :class:`KnowledgeGraphMemory` if enabled."""
        if self.kg is None:
            return []
        return self.kg.query_triples(subject, predicate, object, start_time, end_time)

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
        if tag in self._usage:
            self._usage.pop(tag, None)

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
        if tag in self._usage:
            self._usage.pop(tag, None)

    def search(
        self,
        query: torch.Tensor,
        k: int = 5,
        return_scores: bool = False,
        return_provenance: bool = False,
    ) -> Tuple[torch.Tensor, List[Any]] | Tuple[torch.Tensor, List[Any], List[float], List[Any]]:
        """Retrieve top-k decoded vectors and their metadata.

        When ``return_scores`` or ``return_provenance`` is ``True`` additional
        lists of cosine similarity scores and provenance metadata are returned.
        """
        if isinstance(self.store, AsyncFaissVectorStore):
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.asearch(query, k))
            else:
                return loop.create_task(self.asearch(query, k))
        if self.cache is not None:
            c_vecs, c_meta = self.cache.search(query.detach().cpu().numpy(), k)
        else:
            c_vecs, c_meta = np.empty((0, query.size(-1)), dtype=np.float32), []
        remaining = k - len(c_meta)
        out_vecs = []
        out_meta = []
        if remaining > 0:
            q = self.compressor.encoder(query).detach().cpu().numpy()
            if q.ndim == 2:
                q = q[0]
            comp_vecs, meta = self.store.search(q, remaining)
            if comp_vecs.shape[0] > 0:
                comp_t = torch.from_numpy(comp_vecs)
                decoded = self.compressor.decoder(comp_t).to(query.device)
                out_vecs.append(decoded)
                out_meta.extend(meta)
                if self.cache is not None:
                    self.cache.add(decoded.detach().cpu().numpy(), meta)
        if c_meta:
            out_vecs.insert(0, torch.from_numpy(c_vecs).to(query.device))
            out_meta = c_meta + out_meta
        if not out_vecs:
            empty = torch.empty(0, query.size(-1), device=query.device)
            self.miss_count += 1
            self.query_count += 1
            if (
                self.adaptive_evict
                and self.query_count % self.evict_check_interval == 0
            ):
                self._update_eviction()
            self.last_trace = None
            return empty, []
        vec = torch.cat(out_vecs, dim=0)
        scores = torch.nn.functional.cosine_similarity(
            vec, query.expand_as(vec), dim=1
        ).tolist()
        if self.retrieval_policy is not None:
            order = self.retrieval_policy.rank(out_meta, scores)
            if order:
                vec = vec[order]
                out_meta = [out_meta[i] for i in order]
                scores = [scores[i] for i in order]
        for m in out_meta:
            if m in self._usage:
                self._usage[m] += 1
        self.hit_count += 1
        self.query_count += 1
        if (
            self.adaptive_evict
            and self.query_count % self.evict_check_interval == 0
        ):
            self._update_eviction()
        self.last_trace = {
            "scores": scores,
            "provenance": list(out_meta),
        }
        if return_scores or return_provenance:
            extras: list = []
            if return_scores:
                extras.append(scores)
            if return_provenance:
                extras.append(list(out_meta))
            return (vec, out_meta, *extras)
        return vec, out_meta

    def search_with_kg(
        self, query: torch.Tensor, k: int = 5
    ) -> Tuple[torch.Tensor, List[Any], List[TimedTriple]]:
        """Retrieve vectors and matching knowledge graph triples."""
        vecs, meta = self.search(query, k)
        triples: list[TimedTriple] = []
        if self.kg is not None:
            for m in meta:
                triples.extend(self.kg.query_triples(subject=str(m)))
        return vecs, meta, triples

    async def asearch(
        self, query: torch.Tensor, k: int = 5, return_scores: bool = False, return_provenance: bool = False
    ) -> Tuple[torch.Tensor, List[Any]] | Tuple[torch.Tensor, List[Any], List[float], List[Any]]:
        """Asynchronously retrieve vectors and metadata."""
        if self.cache is not None:
            c_vecs, c_meta = self.cache.search(query.detach().cpu().numpy(), k)
        else:
            c_vecs, c_meta = np.empty((0, query.size(-1)), dtype=np.float32), []
        remaining = k - len(c_meta)
        out_vecs = []
        out_meta = []
        if remaining > 0:
            q = self.compressor.encoder(query).detach().cpu().numpy()
            if q.ndim == 2:
                q = q[0]
            if isinstance(self.store, AsyncFaissVectorStore):
                comp_vecs, meta = await self.store.asearch(q, remaining)
            else:
                comp_vecs, meta = self.store.search(q, remaining)
            if comp_vecs.shape[0] > 0:
                comp_t = torch.from_numpy(comp_vecs)
                decoded = self.compressor.decoder(comp_t).to(query.device)
                out_vecs.append(decoded)
                out_meta.extend(meta)
                if self.cache is not None:
                    self.cache.add(decoded.detach().cpu().numpy(), meta)
        if c_meta:
            out_vecs.insert(0, torch.from_numpy(c_vecs).to(query.device))
            out_meta = c_meta + out_meta
        if not out_vecs:
            empty = torch.empty(0, query.size(-1), device=query.device)
            self.miss_count += 1
            self.query_count += 1
            if (
                self.adaptive_evict
                and self.query_count % self.evict_check_interval == 0
            ):
                self._update_eviction()
            self.last_trace = None
            return empty, []
        vec = torch.cat(out_vecs, dim=0)
        scores = torch.nn.functional.cosine_similarity(
            vec, query.expand_as(vec), dim=1
        ).tolist()
        if self.retrieval_policy is not None:
            order = self.retrieval_policy.rank(out_meta, scores)
            if order:
                vec = vec[order]
                out_meta = [out_meta[i] for i in order]
                scores = [scores[i] for i in order]
        for m in out_meta:
            if m in self._usage:
                self._usage[m] += 1
        self.hit_count += 1
        self.query_count += 1
        if (
            self.adaptive_evict
            and self.query_count % self.evict_check_interval == 0
        ):
            self._update_eviction()
        self.last_trace = {"scores": scores, "provenance": list(out_meta)}
        if return_scores or return_provenance:
            extras = []
            if return_scores:
                extras.append(scores)
            if return_provenance:
                extras.append(list(out_meta))
            return (vec, out_meta, *extras)
        return vec, out_meta

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
            "decay": getattr(self.compressor, "decay", None),
            "evict_limit": self.evict_limit,
            "usage": self._usage,
        }
        torch.save(comp_state, path / "compressor.pt")
        if isinstance(self.store, FaissVectorStore):
            self.store.save(path / "store")
        elif isinstance(self.store, PQVectorStore):
            self.store.save(path / "pq_store")
        elif isinstance(self.store, LocalitySensitiveHashIndex):
            self.store.save(path / "lsh_store")
        elif not isinstance(self.store, HopfieldStore):
            self.store.save(path / "store.npz")
        if self.cache is not None:
            cache_dir = path / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache.save(cache_dir)

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
            "decay": getattr(self.compressor, "decay", None),
            "evict_limit": self.evict_limit,
            "usage": self._usage,
        }
        torch.save(comp_state, path / "compressor.pt")
        if isinstance(self.store, AsyncFaissVectorStore):
            await self.store.save_async(path / "store")
        elif isinstance(self.store, FaissVectorStore):
            self.store.save(path / "store")
        elif isinstance(self.store, PQVectorStore):
            self.store.save(path / "pq_store")
        elif isinstance(self.store, LocalitySensitiveHashIndex):
            self.store.save(path / "lsh_store")
        elif not isinstance(self.store, HopfieldStore):
            self.store.save(path / "store.npz")
        if self.cache is not None:
            cache_dir = path / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache.save(cache_dir)

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
            temporal_decay=state.get("decay"),
        )
        mem.compressor.encoder.load_state_dict(state["encoder"])
        mem.compressor.decoder.load_state_dict(state["decoder"])
        mem.compressor.buffer.data = [t.clone() for t in state["buffer"]]
        mem.compressor.buffer.count = int(state["count"])
        mem._next_id = int(state.get("next_id", 0))
        mem.evict_limit = state.get("evict_limit")
        mem._usage = {int(k): int(v) for k, v in state.get("usage", {}).items()}
        store_dir = path / "store"
        pq_dir = path / "pq_store"
        lsh_dir = path / "lsh_store"
        if lsh_dir.exists():
            mem.store = LocalitySensitiveHashIndex.load(lsh_dir)
        elif pq_dir.exists():
            mem.store = PQVectorStore.load(pq_dir)
        elif store_dir.exists():
            mem.store = FaissVectorStore.load(store_dir)
        elif (path / "store.npz").exists():
            mem.store = VectorStore.load(path / "store.npz")
        else:
            mem.store = HopfieldStore(dim=int(state["compressed_dim"]))
        cache_dir = path / "cache"
        if cache_dir.exists():
            mem.cache = SSDCache.load(cache_dir)
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
            temporal_decay=state.get("decay"),
        )
        mem.compressor.encoder.load_state_dict(state["encoder"])
        mem.compressor.decoder.load_state_dict(state["decoder"])
        mem.compressor.buffer.data = [t.clone() for t in state["buffer"]]
        mem.compressor.buffer.count = int(state["count"])
        mem._next_id = int(state.get("next_id", 0))
        store_dir = path / "store"
        pq_dir = path / "pq_store"
        lsh_dir = path / "lsh_store"
        if lsh_dir.exists():
            mem.store = LocalitySensitiveHashIndex.load(lsh_dir)
        elif pq_dir.exists():
            mem.store = PQVectorStore.load(pq_dir)
        elif store_dir.exists():
            if use_async:
                mem.store = await AsyncFaissVectorStore.load_async(store_dir)
            else:
                mem.store = FaissVectorStore.load(store_dir)
        elif (path / "store.npz").exists():
            mem.store = VectorStore.load(path / "store.npz")
        else:
            mem.store = HopfieldStore(dim=int(state["compressed_dim"]))
        cache_dir = path / "cache"
        if cache_dir.exists():
            mem.cache = SSDCache.load(cache_dir)
        return mem


if _HAS_GRPC:
    class MemoryServer(memory_pb2_grpc.MemoryServiceServicer):
        """gRPC server exposing a ``HierarchicalMemory`` backend."""

        def __init__(
            self,
            memory: HierarchicalMemory,
            address: str = "localhost:50051",
            max_workers: int = 4,
            telemetry: "TelemetryLogger | None" = None,
        ) -> None:
            self.memory = memory
            self.address = address
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
            memory_pb2_grpc.add_MemoryServiceServicer_to_server(self, self.server)
            self.server.add_insecure_port(address)
            self.telemetry = telemetry

        def Push(self, request: memory_pb2.PushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            vec = torch.tensor(request.vector).reshape(1, -1)
            meta = request.metadata if request.metadata else None
            self.memory.add(vec, metadata=[meta])
            if self.telemetry:
                stats = self.telemetry.get_stats()
                stats["push"] = stats.get("push", 0) + 1
                print("telemetry", stats)
            return memory_pb2.PushReply(ok=True)

        def Query(self, request: memory_pb2.QueryRequest, context) -> memory_pb2.QueryReply:  # noqa: N802
            q = torch.tensor(request.vector).reshape(1, -1)
            out, meta = self.memory.search(q, k=int(request.k))
            flat = out.detach().cpu().view(-1).tolist()
            meta = [str(m) for m in meta]
            if self.telemetry:
                stats = self.telemetry.get_stats()
                stats["query"] = stats.get("query", 0) + 1
                print("telemetry", stats)
            return memory_pb2.QueryReply(vectors=flat, metadata=meta)

        def PushBatch(
            self, request: memory_pb2.PushBatchRequest, context
        ) -> memory_pb2.PushReply:  # noqa: N802
            for item in request.items:
                vec = torch.tensor(item.vector).reshape(1, -1)
                meta = item.metadata if item.metadata else None
                self.memory.add(vec, metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def QueryBatch(
            self, request: memory_pb2.QueryBatchRequest, context
        ) -> memory_pb2.QueryBatchReply:  # noqa: N802
            replies = []
            for q in request.items:
                qt = torch.tensor(q.vector).reshape(1, -1)
                out, meta = self.memory.search(qt, k=int(q.k))
                flat = out.detach().cpu().view(-1).tolist()
                meta = [str(m) for m in meta]
                replies.append(memory_pb2.QueryReply(vectors=flat, metadata=meta))
            return memory_pb2.QueryBatchReply(items=replies)

        def start(self) -> None:
            """Start serving requests."""
            if self.telemetry:
                self.telemetry.start()
            self.server.start()

        def stop(self, grace: float = 0) -> None:
            """Stop the server."""
            self.server.stop(grace)
            if self.telemetry:
                self.telemetry.stop()


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


def push_batch_remote(
    address: str, vectors: torch.Tensor, metadata: Iterable[Any] | None = None, timeout: float = 5.0
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


async def push_batch_remote_async(
    address: str, vectors: torch.Tensor, metadata: Iterable[Any] | None = None, timeout: float = 5.0
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
