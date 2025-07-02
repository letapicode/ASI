from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any, List, Tuple

import numpy as np


class PQVectorStore:
    """FAISS IndexIVFPQ-backed vector store."""

    def __init__(
        self,
        dim: int,
        path: str | Path | None = None,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
    ) -> None:
        import faiss

        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
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

    # --------------------------------------------------------------
    def __len__(self) -> int:
        return self.index.ntotal

    # --------------------------------------------------------------
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
        if not self.index.is_trained:
            needed = max(self.nlist, 1 << self.nbits)
            if arr.shape[0] < needed:
                reps = needed // arr.shape[0] + 1
                train_vecs = np.tile(arr, (reps, 1))[:needed]
            else:
                train_vecs = arr
            self.index.train(train_vecs)
        self.index.add(arr)
        self._vectors = np.concatenate([self._vectors, arr], axis=0)
        self._meta.extend(metas)
        if self.path:
            faiss.write_index(self.index, str(self.path / "index.faiss"))
            np.save(self.path / "vectors.npy", self._vectors)
            np.save(self.path / "meta.npy", np.array(self._meta, dtype=object))

    # --------------------------------------------------------------
    def delete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
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
        mask = np.ones(len(self._meta), dtype=bool)
        for i in indices:
            if 0 <= i < len(mask):
                mask[i] = False
        self._vectors = self._vectors[mask]
        self._meta = [m for j, m in enumerate(self._meta) if mask[j]]
        self.index = faiss.IndexIVFPQ(faiss.IndexFlatIP(self.dim), self.dim, self.nlist, self.m, self.nbits)
        if self._vectors.size:
            self.index.train(self._vectors)
            self.index.add(self._vectors)
        if self.path:
            faiss.write_index(self.index, str(self.path / "index.faiss"))
            np.save(self.path / "vectors.npy", self._vectors)
            np.save(self.path / "meta.npy", np.array(self._meta, dtype=object))

    # --------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        if self.index.ntotal == 0:
            return np.empty((0, self.dim), dtype=np.float32), []
        import faiss

        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        self.index.nprobe = min(self.nlist, 8)
        _, idx = self.index.search(q, k)
        idx = idx[0]
        idx = idx[idx >= 0]
        return self._vectors[idx], [self._meta[i] for i in idx]

    # --------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(path / "vectors.npy", self._vectors)
        np.save(path / "meta.npy", np.array(self._meta, dtype=object))
        cfg = {"nlist": self.nlist, "m": self.m, "nbits": self.nbits}
        np.save(path / "config.npy", np.array(cfg, dtype=object))

    # --------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "PQVectorStore":
        import faiss

        path = Path(path)
        cfg_path = path / "config.npy"
        if cfg_path.exists():
            cfg = np.load(cfg_path, allow_pickle=True).item()
        else:
            cfg = {"nlist": 100, "m": 8, "nbits": 8}
        store = cls(
            int(faiss.read_index(str(path / "index.faiss")).d),
            path,
            nlist=int(cfg.get("nlist", 100)),
            m=int(cfg.get("m", 8)),
            nbits=int(cfg.get("nbits", 8)),
        )
        return store


__all__ = ["PQVectorStore"]
