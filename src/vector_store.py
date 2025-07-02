import numpy as np
from typing import Iterable, List, Tuple, Any, Dict
from pathlib import Path

class VectorStore:
    """In-memory vector store with simple top-k retrieval."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: List[np.ndarray] = []
        self._meta: List[Any] = []

    def __len__(self) -> int:
        return sum(v.shape[0] for v in self._vectors)

    def add(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        """Add vectors with optional metadata."""
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self.dim:
            raise ValueError("vector dimension mismatch")
        if metadata is None:
            metas = [None] * arr.shape[0]
        else:
            metas = list(metadata)
            if len(metas) != arr.shape[0]:
                raise ValueError("metadata length mismatch")
        self._vectors.append(arr)
        self._meta.extend(metas)

    def delete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
        """Delete vectors by absolute index or metadata tag."""
        if index is None and tag is None:
            raise ValueError("index or tag must be specified")
        if isinstance(index, Iterable) and not isinstance(index, (bytes, str, bytearray)):
            indices = sorted(int(i) for i in index)
        elif index is not None:
            indices = [int(index)]
        else:
            indices = [i for i, m in enumerate(self._meta) if m == tag]
        if not indices:
            return
        vecs = (
            np.concatenate(self._vectors, axis=0)
            if self._vectors
            else np.empty((0, self.dim), dtype=np.float32)
        )
        mask = np.ones(len(self._meta), dtype=bool)
        for i in indices:
            if 0 <= i < len(mask):
                mask[i] = False
        self._vectors = [vecs[mask]] if mask.any() else []
        self._meta = [m for j, m in enumerate(self._meta) if mask[j]]

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        """Return top-k vectors and metadata by dot-product similarity."""
        if not self._vectors:
            return np.empty((0, self.dim), dtype=np.float32), []
        mat = np.concatenate(self._vectors, axis=0)
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        scores = mat @ q.T  # (n,1)
        idx = np.argsort(scores.ravel())[::-1][:k]
        return mat[idx], [self._meta[i] for i in idx]

    def save(self, path: str | Path) -> None:
        """Persist vectors and metadata to a compressed ``.npz`` file."""
        vecs = (
            np.concatenate(self._vectors, axis=0)
            if self._vectors
            else np.empty((0, self.dim), dtype=np.float32)
        )
        meta = np.array(self._meta, dtype=object)
        np.savez_compressed(path, dim=self.dim, vectors=vecs, meta=meta)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Load vectors and metadata from ``save()`` output."""
        data = np.load(path, allow_pickle=True)
        store = cls(int(data["dim"]))
        vectors = data["vectors"]
        meta = data["meta"].tolist()
        if vectors.size:
            store.add(vectors, metadata=meta)
        return store


class FaissVectorStore:
    """FAISS-backed vector store persisted on disk."""

    def __init__(self, dim: int, path: str | Path | None = None) -> None:
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._meta: List[Any] = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            idx_file = self.path / "index.faiss"
            vec_file = self.path / "vectors.npy"
            meta_file = self.path / "meta.npy"
            if idx_file.exists():
                self.index = faiss.read_index(str(idx_file))
            if vec_file.exists():
                self._vectors = np.load(vec_file)
            if meta_file.exists():
                self._meta = np.load(meta_file, allow_pickle=True).tolist()

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        import faiss

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self.dim:
            raise ValueError("vector dimension mismatch")
        if metadata is None:
            metas = [None] * arr.shape[0]
        else:
            metas = list(metadata)
            if len(metas) != arr.shape[0]:
                raise ValueError("metadata length mismatch")
        self.index.add(arr)
        self._vectors = np.concatenate([self._vectors, arr], axis=0)
        self._meta.extend(metas)
        if self.path:
            faiss.write_index(self.index, str(self.path / "index.faiss"))
            np.save(self.path / "vectors.npy", self._vectors)
            np.save(self.path / "meta.npy", np.array(self._meta, dtype=object))

    def delete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
        """Delete vectors by absolute index or metadata tag."""
        import faiss

        if index is None and tag is None:
            raise ValueError("index or tag must be specified")
        if isinstance(index, Iterable) and not isinstance(index, (bytes, str, bytearray)):
            indices = sorted(int(i) for i in index)
        elif index is not None:
            indices = [int(index)]
        else:
            indices = [i for i, m in enumerate(self._meta) if m == tag]
        if not indices:
            return
        mask = np.ones(self._vectors.shape[0], dtype=bool)
        for i in indices:
            if 0 <= i < len(mask):
                mask[i] = False
        self._vectors = self._vectors[mask]
        self._meta = [m for j, m in enumerate(self._meta) if mask[j]]
        self.index = faiss.IndexFlatIP(self.dim)
        if self._vectors.size:
            self.index.add(self._vectors)
        if self.path:
            faiss.write_index(self.index, str(self.path / "index.faiss"))
            np.save(self.path / "vectors.npy", self._vectors)
            np.save(self.path / "meta.npy", np.array(self._meta, dtype=object))

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        if self.index.ntotal == 0:
            return np.empty((0, self.dim), dtype=np.float32), []
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        _, idx = self.index.search(q, k)
        idx = idx[0]
        idx = idx[idx >= 0]
        return self._vectors[idx], [self._meta[i] for i in idx]

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(path / "vectors.npy", self._vectors)
        np.save(path / "meta.npy", np.array(self._meta, dtype=object))

    @classmethod
    def load(cls, path: str | Path) -> "FaissVectorStore":
        import faiss

        path = Path(path)
        store = cls(int(faiss.read_index(str(path / "index.faiss")).d), path)
        return store


class LocalitySensitiveHashIndex:
    """Approximate vector store using LSH buckets."""

    def __init__(self, dim: int, num_planes: int = 16) -> None:
        self.dim = dim
        self.num_planes = num_planes
        self.hyperplanes = np.random.randn(num_planes, dim).astype(np.float32)
        self.buckets: Dict[int, list[int]] = {}
        self.vectors: list[np.ndarray] = []
        self.meta: list[Any] = []

    def __len__(self) -> int:
        return len(self.meta)

    def _hash(self, vec: np.ndarray) -> int:
        signs = (vec @ self.hyperplanes.T) > 0
        h = 0
        for i, s in enumerate(signs):
            if s:
                h |= 1 << i
        return int(h)

    def add(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self.dim:
            raise ValueError("vector dimension mismatch")
        metas = list(metadata) if metadata is not None else [None] * arr.shape[0]
        if len(metas) != arr.shape[0]:
            raise ValueError("metadata length mismatch")
        start_idx = len(self.vectors)
        for i, vec in enumerate(arr):
            idx = start_idx + i
            h = self._hash(vec)
            self.buckets.setdefault(h, []).append(idx)
            self.vectors.append(vec)
        self.meta.extend(metas)

    def delete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
        if index is None and tag is None:
            raise ValueError("index or tag must be specified")
        if isinstance(index, Iterable) and not isinstance(index, (bytes, str, bytearray)):
            indices = sorted(int(i) for i in index)
        elif index is not None:
            indices = [int(index)]
        else:
            indices = [i for i, m in enumerate(self.meta) if m == tag]
        if not indices:
            return
        mask = np.ones(len(self.vectors), dtype=bool)
        for i in indices:
            if 0 <= i < len(mask):
                mask[i] = False
        self.vectors = [v for j, v in enumerate(self.vectors) if mask[j]]
        self.meta = [m for j, m in enumerate(self.meta) if mask[j]]
        self.buckets.clear()
        for idx, vec in enumerate(self.vectors):
            h = self._hash(vec)
            self.buckets.setdefault(h, []).append(idx)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        if len(self.vectors) == 0:
            return np.empty((0, self.dim), dtype=np.float32), []
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        h = self._hash(q[0])
        candidates = self.buckets.get(h, [])
        if not candidates:
            # fall back to linear search
            mat = np.asarray(self.vectors)
            scores = mat @ q.T
            idx = np.argsort(scores.ravel())[::-1][:k]
            return mat[idx], [self.meta[i] for i in idx]
        mat = np.asarray([self.vectors[i] for i in candidates])
        scores = mat @ q.T
        idx = np.argsort(scores.ravel())[::-1][:k]
        selected = [candidates[i] for i in idx]
        return mat[idx], [self.meta[i] for i in selected]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "lsh.npz",
            dim=self.dim,
            planes=self.hyperplanes,
            vectors=np.asarray(self.vectors, dtype=np.float32),
            meta=np.array(self.meta, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LocalitySensitiveHashIndex":
        path = Path(path)
        data = np.load(path / "lsh.npz", allow_pickle=True)
        store = cls(int(data["dim"]), data["planes"].shape[0])
        store.hyperplanes = data["planes"]
        store.vectors = data["vectors"].tolist()
        store.meta = data["meta"].tolist()
        store.buckets.clear()
        for idx, vec in enumerate(store.vectors):
            h = store._hash(vec)
            store.buckets.setdefault(h, []).append(idx)
        return store


__all__ = [
    "VectorStore",
    "FaissVectorStore",
    "LocalitySensitiveHashIndex",
]
