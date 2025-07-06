import time
from typing import Iterable, Any, List, Tuple

import numpy as np


class EphemeralVectorStore:
    """In-memory vector store that drops old entries after ``ttl`` seconds."""

    def __init__(self, dim: int, ttl: float = 60.0) -> None:
        self.dim = dim
        self.ttl = float(ttl)
        self._vectors: List[np.ndarray] = []
        self._meta: List[Any] = []
        self._time: List[float] = []

    def __len__(self) -> int:
        self.cleanup_expired()
        return len(self._meta)

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
        now = time.time()
        self._vectors.extend(arr)
        self._meta.extend(metas)
        self._time.extend([now] * arr.shape[0])
        self.cleanup_expired()

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        self.cleanup_expired()
        if not self._vectors:
            return np.empty((0, self.dim), dtype=np.float32), []
        mat = np.vstack(self._vectors)
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        scores = mat @ q.T
        idx = np.argsort(scores.ravel())[::-1][:k]
        return mat[idx], [self._meta[i] for i in idx]

    def cleanup_expired(self) -> None:
        """Remove entries older than the TTL."""
        if not self._time:
            return
        cutoff = time.time() - self.ttl
        mask = [t >= cutoff for t in self._time]
        if all(mask):
            return
        self._vectors = [v for v, m in zip(self._vectors, mask) if m]
        self._meta = [m for m, keep in zip(self._meta, mask) if keep]
        self._time = [t for t in self._time if t >= cutoff]


__all__ = ["EphemeralVectorStore"]

