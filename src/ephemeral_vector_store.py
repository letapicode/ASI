import time
from typing import Iterable, Any, List, Tuple

import numpy as np

from .vector_store import VectorStore


class EphemeralVectorStore(VectorStore):
    """VectorStore that drops entries older than ``ttl`` seconds."""

    def __init__(self, dim: int, ttl: float = 60.0) -> None:
        super().__init__(dim)
        self.ttl = float(ttl)
        self._time: List[float] = []

    def __len__(self) -> int:
        self.cleanup_expired()
        return super().__len__()

    def add(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        super().add(arr, metadata=metadata)
        now = time.time()
        self._time.extend([now] * arr.shape[0])
        self.cleanup_expired()

    def search(
        self, query: np.ndarray, k: int = 5, *, quantum: bool = False
    ) -> Tuple[np.ndarray, List[Any]]:
        self.cleanup_expired()
        return super().search(query, k, quantum=quantum)

    def delete(
        self, index: int | Iterable[int] | None = None, tag: Any | None = None
    ) -> None:
        self.cleanup_expired()
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
        self._meta_map = {m: i for i, m in enumerate(self._meta)}
        self._time = [t for j, t in enumerate(self._time) if mask[j]]

    def cleanup_expired(self) -> None:
        """Remove entries older than ``ttl`` seconds."""
        if not self._time:
            return
        cutoff = time.time() - self.ttl
        mask = np.array([t >= cutoff for t in self._time], dtype=bool)
        if mask.all():
            return
        vecs = (
            np.concatenate(self._vectors, axis=0)
            if self._vectors
            else np.empty((0, self.dim), dtype=np.float32)
        )
        vecs = vecs[mask]
        self._vectors = [vecs] if vecs.size else []
        self._meta = [m for j, m in enumerate(self._meta) if mask[j]]
        self._meta_map = {m: i for i, m in enumerate(self._meta)}
        self._time = [t for j, t in enumerate(self._time) if mask[j]]


__all__ = ["EphemeralVectorStore"]

