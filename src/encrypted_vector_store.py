import os
import io
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .vector_store import VectorStore


class EncryptedVectorStore(VectorStore):
    """Vector store that encrypts embeddings with AES-GCM when persisted."""

    def __init__(self, dim: int, key: bytes) -> None:
        super().__init__(dim)
        if len(key) not in (16, 24, 32):
            raise ValueError("key must be 16, 24, or 32 bytes")
        self._key = key

    # ------------------------------------------------------------------
    @property
    def key(self) -> bytes:
        return self._key

    def set_key(self, key: bytes) -> None:
        if len(key) not in (16, 24, 32):
            raise ValueError("key must be 16, 24, or 32 bytes")
        self._key = key

    def rotate_key(self, new_key: bytes, path: str | Path | None = None) -> None:
        """Update encryption key and optionally re-save the store."""
        self.set_key(new_key)
        if path is not None:
            self.save(path)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:  # type: ignore[override]
        """Persist encrypted vectors and metadata to ``path``."""
        vecs = (
            np.concatenate(self._vectors, axis=0)
            if self._vectors
            else np.empty((0, self.dim), dtype=np.float32)
        )
        meta = np.array(self._meta, dtype=object)
        buf = io.BytesIO()
        np.savez_compressed(buf, dim=self.dim, vectors=vecs, meta=meta)
        data = buf.getvalue()
        aes = AESGCM(self._key)
        nonce = os.urandom(12)
        enc = aes.encrypt(nonce, data, None)
        with open(path, "wb") as f:
            f.write(nonce + enc)

    @classmethod
    def load(cls, path: str | Path, key: bytes) -> "EncryptedVectorStore":
        """Load an encrypted vector store from ``path``."""
        if len(key) not in (16, 24, 32):
            raise ValueError("key must be 16, 24, or 32 bytes")
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


__all__ = ["EncryptedVectorStore"]
