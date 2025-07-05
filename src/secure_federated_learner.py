from __future__ import annotations

from typing import Iterable

from .zk_verifier import ZKVerifier

import torch

class SecureFederatedLearner:
    """Aggregate encrypted gradients from remote peers."""

    def __init__(
        self,
        key: int = 0,
        *,
        require_proof: bool = False,
        zk: ZKVerifier | None = None,
    ) -> None:
        self.key = key
        self.require_proof = require_proof
        self.zk = zk or ZKVerifier()

    def encrypt(self, grad: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(self.key)
        noise = torch.randn_like(grad)
        self._last_noise = noise
        return grad + noise

    def decrypt(self, grad: torch.Tensor) -> torch.Tensor:
        return grad - getattr(self, "_last_noise", torch.zeros_like(grad))

    def aggregate(
        self, grads: Iterable[torch.Tensor], proofs: Iterable[str] | None = None
    ) -> torch.Tensor:
        grads = list(grads)
        if not grads:
            raise ValueError("no gradients")
        if self.require_proof:
            if proofs is None:
                raise ValueError("proofs required")
            proofs = list(proofs)
            if len(proofs) != len(grads):
                raise ValueError("proof count mismatch")
            for g, p in zip(grads, proofs):
                if not self.zk.verify_proof(g, p):
                    raise ValueError("invalid proof")
        agg = torch.stack(grads).mean(dim=0)
        return agg

__all__ = ["SecureFederatedLearner"]
