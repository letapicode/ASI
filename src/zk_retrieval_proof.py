from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ZKRetrievalProof:
    """Simple hash-based proof for retrieved vectors."""

    digest: str

    @classmethod
    def generate(
        cls, vectors: np.ndarray, metadata: Iterable[str] | None = None
    ) -> "ZKRetrievalProof":
        arr = np.asarray(vectors, dtype=np.float32)
        arr = np.round(arr, decimals=6)
        data = arr.tobytes()
        if metadata is not None:
            data += "|".join(str(m) for m in metadata).encode()
        digest = hashlib.sha256(data).hexdigest()
        return cls(digest)

    def verify(self, vectors: np.ndarray, metadata: Iterable[str] | None = None) -> bool:
        arr = np.asarray(vectors, dtype=np.float32)
        arr = np.round(arr, decimals=6)
        data = arr.tobytes()
        if metadata is not None:
            data += "|".join(str(m) for m in metadata).encode()
        return hashlib.sha256(data).hexdigest() == self.digest


__all__ = ["ZKRetrievalProof"]
