from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


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


__all__ = ["ZKGradientProof"]
