from __future__ import annotations

import hashlib
from dataclasses import dataclass

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


__all__ = ["RetrievalProof"]
