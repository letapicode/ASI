from __future__ import annotations

import hashlib

import torch


class ZKVerifier:
    """Generate and verify simple proofs for tensors."""

    def generate_proof(self, grad: torch.Tensor) -> str:
        data = grad.detach().cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def verify_proof(self, grad: torch.Tensor, proof: str) -> bool:
        return self.generate_proof(grad) == proof


__all__ = ["ZKVerifier"]

