from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class RetrievalProof:
    """Hash-based proof of vector integrity."""

    digest: str

    @classmethod
    def generate(cls, vector: torch.Tensor) -> "RetrievalProof":
        data = vector.detach().cpu().numpy().tobytes()
        digest = hashlib.sha256(data).hexdigest()
        return cls(digest)

    def verify(self, vector: torch.Tensor) -> bool:
        data = vector.detach().cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest() == self.digest


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


@dataclass
class ZKGradientProof:
    """Simple hash-based proof of gradient correctness."""

    digest: str

    @classmethod
    def generate(cls, grad: torch.Tensor) -> "ZKGradientProof":
        data = grad.detach().cpu().numpy().tobytes()
        digest = hashlib.sha256(data).hexdigest()
        return cls(digest)

    def verify(self, grad: torch.Tensor) -> bool:
        data = grad.detach().cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest() == self.digest


class ZKVerifier:
    """Generate and verify simple proofs for tensors."""

    def generate_proof(self, grad: torch.Tensor) -> str:
        data = grad.detach().cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def verify_proof(self, grad: torch.Tensor, proof: str) -> bool:
        return self.generate_proof(grad) == proof


__all__ = [
    "RetrievalProof",
    "ZKRetrievalProof",
    "ZKGradientProof",
    "ZKVerifier",
]
