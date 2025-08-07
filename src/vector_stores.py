"""Unified vector store implementations."""

from __future__ import annotations

import asyncio
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Iterable, Any, List, Tuple, Dict

import numpy as np

try:  # optional quantum retrieval
    from .quantum_sampling import amplify_search as _amplify_search
except Exception:  # pragma: no cover - optional dependency
    _amplify_search = None


class BaseVectorStore:
    """Mixin providing basic vector store operations."""

    dim: int
    _vectors: List[np.ndarray]
    _meta: List[Any]
    _meta_map: Dict[Any, int]

    def __len__(self) -> int:
        return sum(v.shape[0] for v in self._vectors)

    def _all_vectors(self) -> np.ndarray:
        return (
            np.concatenate(self._vectors, axis=0)
            if self._vectors
            else np.empty((0, self.dim), dtype=np.float32)
        )

    def add(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
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
        self._add_vectors(arr, metas)

    def _add_vectors(self, arr: np.ndarray, metas: List[Any]) -> None:
        self._vectors.append(arr)
        start = len(self._meta)
        self._meta.extend(metas)
        for i, m in enumerate(metas):
            self._meta_map[m] = start + i

    def delete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
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
        self._delete_indices(indices)

    def _delete_indices(self, indices: list[int]) -> None:
        vecs = self._all_vectors()
        mask = np.ones(len(self._meta), dtype=bool)
        for i in indices:
            if 0 <= i < len(mask):
                mask[i] = False
        self._apply_mask(vecs, mask)

    def _apply_mask(self, vecs: np.ndarray, mask: np.ndarray) -> None:
        self._vectors = [vecs[mask]] if mask.any() else []
        self._meta = [m for j, m in enumerate(self._meta) if mask[j]]
        self._meta_map = {m: i for i, m in enumerate(self._meta)}

    def search(self, query: np.ndarray, k: int = 5, *, quantum: bool = False) -> Tuple[np.ndarray, List[Any]]:
        if not self._vectors:
            return np.empty((0, self.dim), dtype=np.float32), []
        mat = self._all_vectors()
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        time.sleep(0.005)
        scores = mat @ q.T
        if quantum and _amplify_search is not None:
            idx = _amplify_search(scores.ravel(), k)
        else:
            idx = np.argsort(scores.ravel())[::-1][:k]
        return mat[idx], [self._meta[i] for i in idx]

    def _encode_hypothetical(self, query: np.ndarray) -> np.ndarray:
        return np.asarray(query, dtype=np.float32)

    def hyde_search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        q = np.asarray(query, dtype=np.float32)
        hyp = self._encode_hypothetical(q)
        blend = (q + hyp) / 2.0
        return self.search(blend, k)


class VectorStore(BaseVectorStore):
    """In-memory vector store with simple top-k retrieval."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = []
        self._meta = []
        self._meta_map = {}

    def save(self, path: str | Path) -> None:
        vecs = self._all_vectors()
        meta = np.array(self._meta, dtype=object)
        np.savez_compressed(path, dim=self.dim, vectors=vecs, meta=meta)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
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
        self._meta_map: Dict[Any, int] = {}
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
                self._meta_map = {m: i for i, m in enumerate(self._meta)}

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
        start = len(self._meta)
        self._meta.extend(metas)
        for i, m in enumerate(metas):
            self._meta_map[m] = start + i
        if self.path:
            faiss.write_index(self.index, str(self.path / "index.faiss"))
            np.save(self.path / "vectors.npy", self._vectors)
            np.save(self.path / "meta.npy", np.array(self._meta, dtype=object))

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
        mask = np.ones(self._vectors.shape[0], dtype=bool)
        for i in indices:
            if 0 <= i < len(mask):
                mask[i] = False
        self._vectors = self._vectors[mask]
        self._meta = [m for j, m in enumerate(self._meta) if mask[j]]
        self._meta_map = {m: i for i, m in enumerate(self._meta)}
        self.index = faiss.IndexFlatIP(self.dim)
        if self._vectors.size:
            self.index.add(self._vectors)
        if self.path:
            faiss.write_index(self.index, str(self.path / "index.faiss"))
            np.save(self.path / "vectors.npy", self._vectors)
            np.save(self.path / "meta.npy", np.array(self._meta, dtype=object))

    def search(self, query: np.ndarray, k: int = 5, *, quantum: bool = False) -> Tuple[np.ndarray, List[Any]]:
        if self.index.ntotal == 0:
            return np.empty((0, self.dim), dtype=np.float32), []
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        if quantum and _amplify_search is not None:
            scores = self._vectors @ q.T
            idx = _amplify_search(scores.ravel(), k)
        else:
            _, idx = self.index.search(q, k)
            idx = idx[0]
            idx = idx[idx >= 0]
        return self._vectors[idx], [self._meta[i] for i in idx]

    def _encode_hypothetical(self, query: np.ndarray) -> np.ndarray:
        return np.asarray(query, dtype=np.float32)

    def hyde_search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        q = np.asarray(query, dtype=np.float32)
        hyp = self._encode_hypothetical(q)
        blend = (q + hyp) / 2.0
        return self.search(blend, k)

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


class LocalitySensitiveHashIndex(BaseVectorStore):
    """Approximate vector store using LSH buckets."""

    def __init__(self, dim: int, num_planes: int = 16) -> None:
        self.dim = dim
        self.num_planes = num_planes
        self.hyperplanes = np.random.randn(num_planes, dim).astype(np.float32)
        self.buckets: Dict[int, list[int]] = {}
        self._vectors: list[np.ndarray] = []
        self._meta: list[Any] = []
        self._meta_map: Dict[Any, int] = {}

    def _hash(self, vec: np.ndarray) -> int:
        signs = (vec @ self.hyperplanes.T) > 0
        h = 0
        for i, s in enumerate(signs):
            if s:
                h |= 1 << i
        return int(h)

    def _add_vectors(self, arr: np.ndarray, metas: List[Any]) -> None:
        start_idx = len(self._vectors)
        for i, vec in enumerate(arr):
            idx = start_idx + i
            h = self._hash(vec)
            self.buckets.setdefault(h, []).append(idx)
            self._vectors.append(vec)
        self._meta.extend(metas)
        for i, m in enumerate(metas):
            self._meta_map[m] = start_idx + i

    def _apply_mask(self, vecs: np.ndarray, mask: np.ndarray) -> None:
        self._vectors = [v for j, v in enumerate(self._vectors) if mask[j]]
        self._meta = [m for j, m in enumerate(self._meta) if mask[j]]
        self._meta_map = {m: i for i, m in enumerate(self._meta)}
        self.buckets.clear()
        for idx, vec in enumerate(self._vectors):
            h = self._hash(vec)
            self.buckets.setdefault(h, []).append(idx)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        if len(self._vectors) == 0:
            return np.empty((0, self.dim), dtype=np.float32), []
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        h = self._hash(q[0])
        candidates = self.buckets.get(h, [])
        if not candidates:
            mat = np.asarray(self._vectors)
            scores = mat @ q.T
            idx = np.argsort(scores.ravel())[::-1][:k]
            return mat[idx], [self._meta[i] for i in idx]
        mat = np.asarray([self._vectors[i] for i in candidates])
        scores = mat @ q.T
        idx = np.argsort(scores.ravel())[::-1][:k]
        selected = [candidates[i] for i in idx]
        return mat[idx], [self._meta[i] for i in selected]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "lsh.npz",
            dim=self.dim,
            planes=self.hyperplanes,
            vectors=np.asarray(self._vectors, dtype=np.float32),
            meta=np.array(self._meta, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LocalitySensitiveHashIndex":
        path = Path(path)
        data = np.load(path / "lsh.npz", allow_pickle=True)
        store = cls(int(data["dim"]), data["planes"].shape[0])
        store.hyperplanes = data["planes"]
        store._vectors = data["vectors"].tolist()
        store._meta = data["meta"].tolist()
        store._meta_map = {m: i for i, m in enumerate(store._meta)}
        for idx, vec in enumerate(store._vectors):
            h = store._hash(vec)
            store.buckets.setdefault(h, []).append(idx)
        return store


class EncryptedVectorStore(VectorStore):
    """Vector store that encrypts embeddings with AES-GCM when persisted."""

    def __init__(self, dim: int, key: bytes) -> None:
        super().__init__(dim)
        if len(key) not in (16, 24, 32):
            raise ValueError("key must be 16, 24, or 32 bytes")
        self._key = key

    @property
    def key(self) -> bytes:
        return self._key

    def set_key(self, key: bytes) -> None:
        if len(key) not in (16, 24, 32):
            raise ValueError("key must be 16, 24, or 32 bytes")
        self._key = key

    def rotate_key(self, new_key: bytes, path: str | Path | None = None) -> None:
        self.set_key(new_key)
        if path is not None:
            self.save(path)

    def save(self, path: str | Path) -> None:  # type: ignore[override]
        vecs = self._all_vectors()
        meta = np.array(self._meta, dtype=object)
        buf = io.BytesIO()
        np.savez_compressed(buf, dim=self.dim, vectors=vecs, meta=meta)
        data = buf.getvalue()
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        aes = AESGCM(self._key)
        nonce = os.urandom(12)
        enc = aes.encrypt(nonce, data, None)
        with open(path, "wb") as f:
            f.write(nonce + enc)

    @classmethod
    def load(cls, path: str | Path, key: bytes) -> "EncryptedVectorStore":
        if len(key) not in (16, 24, 32):
            raise ValueError("key must be 16, 24, or 32 bytes")
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        with open(path, "rb") as f:
            blob = f.read()
        nonce, enc = blob[:12], blob[12:]
        aes = AESGCM(key)
        data = aes.decrypt(nonce, enc, None)
        buf = io.BytesIO(data)
        npz = np.load(buf, allow_pickle=True)
        store = cls(int(npz["dim"]), key)
        vectors = npz["vectors"]
        meta = npz["meta"].tolist()
        if vectors.size:
            store.add(vectors, metadata=meta)
        return store


class EphemeralVectorStore(VectorStore):
    """VectorStore that drops entries older than ``ttl`` seconds."""

    def __init__(self, dim: int, ttl: float = 60.0) -> None:
        super().__init__(dim)
        self.ttl = float(ttl)
        self._time: List[float] = []

    def __len__(self) -> int:
        self.cleanup_expired()
        return super().__len__()

    def _add_vectors(self, arr: np.ndarray, metas: List[Any]) -> None:  # type: ignore[override]
        super()._add_vectors(arr, metas)
        now = time.time()
        self._time.extend([now] * arr.shape[0])
        self.cleanup_expired()

    def _apply_mask(self, vecs: np.ndarray, mask: np.ndarray) -> None:  # type: ignore[override]
        super()._apply_mask(vecs, mask)
        self._time = [t for j, t in enumerate(self._time) if mask[j]]

    def search(self, query: np.ndarray, k: int = 5, *, quantum: bool = False) -> Tuple[np.ndarray, List[Any]]:
        self.cleanup_expired()
        return super().search(query, k, quantum=quantum)

    def cleanup_expired(self) -> None:
        if not self._time:
            return
        cutoff = time.time() - self.ttl
        mask = np.array([t >= cutoff for t in self._time], dtype=bool)
        if mask.all():
            return
        vecs = self._all_vectors()[mask]
        self._apply_mask(vecs, mask)


class HolographicVectorStore:
    """Vector store using holographic reduced representations."""

    def __init__(self, dim: int, path: str | Path | None = None) -> None:
        self.dim = dim
        self.store = VectorStore(dim)
        self.keys = {
            "text": np.random.randn(dim).astype(np.float32),
            "image": np.random.randn(dim).astype(np.float32),
            "audio": np.random.randn(dim).astype(np.float32),
        }
        self._fkeys = {k: np.fft.rfft(v) for k, v in self.keys.items()}
        self.path = Path(path) if path else None
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            key_file = self.path / "hrr_keys.npz"
            store_file = self.path / "store.npz"
            if key_file.exists():
                data = np.load(key_file)
                for k in self.keys:
                    if k in data:
                        self.keys[k] = data[k]
                self._fkeys = {k: np.fft.rfft(self.keys[k]) for k in self.keys}
            if store_file.exists():
                self.store = VectorStore.load(store_file)

    def __len__(self) -> int:
        return len(self.store)

    def encode(
        self,
        text: np.ndarray | None = None,
        image: np.ndarray | None = None,
        audio: np.ndarray | None = None,
    ) -> np.ndarray:
        out_f = np.zeros_like(self._fkeys["text"], dtype=np.complex64)
        if text is not None:
            out_f += np.fft.rfft(text.astype(np.float32)) * self._fkeys["text"]
        if image is not None:
            out_f += np.fft.rfft(image.astype(np.float32)) * self._fkeys["image"]
        if audio is not None:
            out_f += np.fft.rfft(audio.astype(np.float32)) * self._fkeys["audio"]
        out = np.fft.irfft(out_f, n=self.dim)
        return out.astype(np.float32)

    def encode_batch(
        self,
        text: np.ndarray | None = None,
        image: np.ndarray | None = None,
        audio: np.ndarray | None = None,
    ) -> np.ndarray:
        if text is not None:
            n = text.shape[0]
        elif image is not None:
            n = image.shape[0]
        elif audio is not None:
            n = audio.shape[0]
        else:
            return np.empty((0, self.dim), dtype=np.float32)
        out_f = np.zeros((n, self._fkeys["text"].shape[0]), dtype=np.complex64)
        if text is not None:
            out_f += np.fft.rfft(text.astype(np.float32), axis=1) * self._fkeys["text"]
        if image is not None:
            out_f += np.fft.rfft(image.astype(np.float32), axis=1) * self._fkeys["image"]
        if audio is not None:
            out_f += np.fft.rfft(audio.astype(np.float32), axis=1) * self._fkeys["audio"]
        out = np.fft.irfft(out_f, n=self.dim, axis=1)
        return out.astype(np.float32)

    def decode(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vf = np.fft.rfft(vec.astype(np.float32))
        text = np.fft.irfft(vf * np.conj(self._fkeys["text"]), n=self.dim)
        image = np.fft.irfft(vf * np.conj(self._fkeys["image"]), n=self.dim)
        audio = np.fft.irfft(vf * np.conj(self._fkeys["audio"]), n=self.dim)
        return text.astype(np.float32), image.astype(np.float32), audio.astype(np.float32)

    def add(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        self.store.add(vectors, metadata)

    def add_modalities(
        self,
        text: np.ndarray,
        image: np.ndarray,
        audio: np.ndarray,
        metadata: Iterable[Any] | None = None,
    ) -> None:
        arr = self.encode_batch(text, image, audio)
        self.add(arr, metadata)

    def delete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
        self.store.delete(index=index, tag=tag)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        return self.store.search(query, k)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.store.save(path / "store.npz")
        np.savez(path / "hrr_keys.npz", **self.keys)

    @classmethod
    def load(cls, path: str | Path) -> "HolographicVectorStore":
        path = Path(path)
        key_file = path / "hrr_keys.npz"
        store_file = path / "store.npz"
        if not store_file.exists():
            raise FileNotFoundError(store_file)
        store = cls(int(np.load(key_file)["text"].shape[0]), path)
        store.store = VectorStore.load(store_file)
        return store


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
        self._meta_map: Dict[Any, int] = {}
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
                self._meta_map = {m: i for i, m in enumerate(self._meta)}

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
        start = len(self._meta)
        self._meta.extend(metas)
        for i, m in enumerate(metas):
            self._meta_map[m] = start + i
        if self.path:
            faiss.write_index(self.index, str(self.path / "index.faiss"))
            np.save(self.path / "vectors.npy", self._vectors)
            np.save(self.path / "meta.npy", np.array(self._meta, dtype=object))

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
        self._meta_map = {m: i for i, m in enumerate(self._meta)}
        self.index = faiss.IndexIVFPQ(faiss.IndexFlatIP(self.dim), self.dim, self.nlist, self.m, self.nbits)
        if self._vectors.size:
            self.index.train(self._vectors)
            self.index.add(self._vectors)
        if self.path:
            faiss.write_index(self.index, str(self.path / "index.faiss"))
            np.save(self.path / "vectors.npy", self._vectors)
            np.save(self.path / "meta.npy", np.array(self._meta, dtype=object))

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        if self.index.ntotal == 0:
            return np.empty((0, self.dim), dtype=np.float32), []
        import faiss

        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        if self.index.ntotal <= self.nlist:
            scores = self._vectors @ q.T
            idx = np.argsort(scores.ravel())[::-1][:k]
        else:
            self.index.nprobe = min(self.nlist, 8)
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
        cfg = {"nlist": self.nlist, "m": self.m, "nbits": self.nbits}
        np.save(path / "config.npy", np.array(cfg, dtype=object))

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


class AsyncFaissVectorStore(FaissVectorStore):
    """FAISS vector store with async add/search using threads."""

    def __init__(self, dim: int, path: str | Path | None = None, workers: int = 2) -> None:
        super().__init__(dim=dim, path=path)
        self._executor = ThreadPoolExecutor(max_workers=workers)

    def add_async(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> Future:
        return self._executor.submit(super().add, vectors, metadata)

    def search_async(self, query: np.ndarray, k: int = 5) -> Future:
        return self._executor.submit(super().search, query, k)

    def hyde_search_async(self, query: np.ndarray, k: int = 5) -> Future:
        return self._executor.submit(super().hyde_search, query, k)

    async def ahyde_search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, list[Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, super().hyde_search, query, k)

    async def save_async(self, path: str | Path) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, super().save, path)

    @classmethod
    async def load_async(cls, path: str | Path) -> "AsyncFaissVectorStore":
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, cls.load, path)

    async def aadd(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, super().add, vectors, metadata)

    async def asearch(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, list[Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, super().search, query, k)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "AsyncFaissVectorStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aenter__(self) -> "AsyncFaissVectorStore":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "VectorStore",
    "FaissVectorStore",
    "LocalitySensitiveHashIndex",
    "EncryptedVectorStore",
    "EphemeralVectorStore",
    "HolographicVectorStore",
    "PQVectorStore",
    "AsyncFaissVectorStore",
]
