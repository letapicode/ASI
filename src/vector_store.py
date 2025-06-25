import numpy as np
from typing import Iterable, List, Tuple, Any

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

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        """Return top-k vectors and metadata by dot-product similarity."""
        if not self._vectors:
            return np.empty((0, self.dim), dtype=np.float32), []
        mat = np.concatenate(self._vectors, axis=0)
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        scores = mat @ q.T  # (n,1)
        idx = np.argsort(scores.ravel())[::-1][:k]
        return mat[idx], [self._meta[i] for i in idx]

    def save(self, path: str) -> None:
        """Persist the vector store to ``path`` using ``numpy.savez``."""
        if self._vectors:
            mat = np.concatenate(self._vectors, axis=0)
        else:
            mat = np.empty((0, self.dim), dtype=np.float32)
        meta = np.array(self._meta, dtype=object)
        np.savez(path, vectors=mat, meta=meta)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a vector store saved with :meth:`save`."""
        data = np.load(path, allow_pickle=True)
        vectors = data["vectors"]
        meta = data["meta"].tolist()
        store = cls(vectors.shape[1])
        if vectors.size:
            store._vectors = [vectors.astype(np.float32)]
            store._meta = meta
        return store
