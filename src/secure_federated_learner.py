from __future__ import annotations

from typing import Iterable, List, Tuple

from .zk_gradient_proof import ZKGradientProof

import torch

class SecureFederatedLearner:
    """Aggregate encrypted gradients from remote peers."""

    def __init__(self, key: int = 0) -> None:
        self.key = key

    def encrypt(
        self, grad: torch.Tensor, with_proof: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, ZKGradientProof]:
        """Encrypt a gradient and optionally return a zero-knowledge proof."""
        torch.manual_seed(self.key)
        noise = torch.randn_like(grad)
        self._last_noise = noise
        enc = grad + noise
        if with_proof:
            return enc, ZKGradientProof.generate(grad)
        return enc

    def decrypt(
        self, grad: torch.Tensor, proof: ZKGradientProof | None = None
    ) -> torch.Tensor:
        """Decrypt a gradient and verify it if a proof is supplied."""
        dec = grad - getattr(self, "_last_noise", torch.zeros_like(grad))
        if proof and not proof.verify(dec):
            raise ValueError("invalid gradient proof")
        return dec

    def aggregate(self, grads: Iterable[torch.Tensor]) -> torch.Tensor:
        grads = list(grads)
        if not grads:
            raise ValueError("no gradients")
        agg = torch.stack(grads).mean(dim=0)
        return agg

__all__ = ["SecureFederatedLearner"]
