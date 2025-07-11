from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any, List

import numpy as np

from .vector_stores import PQVectorStore


class IncrementalPQIndexer:
    """Manage a sharded PQVectorStore with incremental updates."""

    def __init__(self, dim: int, base_dir: str | Path, shard_size: int = 10000) -> None:
        self.dim = dim
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.shards: List[PQVectorStore] = []
        self._load_shards()

    # --------------------------------------------------------------
    def _load_shards(self) -> None:
        for shard_dir in sorted(self.base_dir.glob("shard_*")):
            self.shards.append(PQVectorStore.load(shard_dir))

    # --------------------------------------------------------------
    def _current_shard(self) -> PQVectorStore:
        if not self.shards or len(self.shards[-1]) >= self.shard_size:
            shard_dir = self.base_dir / f"shard_{len(self.shards)}"
            shard = PQVectorStore(dim=self.dim, path=shard_dir)
            self.shards.append(shard)
        return self.shards[-1]

    # --------------------------------------------------------------
    def add(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        shard = self._current_shard()
        shard.add(vectors, metadata)

    # --------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 5) -> tuple[np.ndarray, List[Any]]:
        all_vecs: list[np.ndarray] = []
        all_meta: list[Any] = []
        for shard in self.shards:
            vecs, meta = shard.search(query, k)
            if vecs.size:
                all_vecs.append(vecs)
                all_meta.extend(meta)
        if not all_meta:
            return np.empty((0, self.dim), dtype=np.float32), []
        mat = np.concatenate(all_vecs, axis=0)
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores = mat @ q.T
        idx = np.argsort(scores.ravel())[::-1][:k]
        return mat[idx], [all_meta[i] for i in idx]

    # --------------------------------------------------------------
    def save(self) -> None:
        for shard in self.shards:
            shard.save(shard.path)

