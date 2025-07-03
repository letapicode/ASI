from __future__ import annotations

from typing import Iterable, List

import torch

class SecureFederatedLearner:
    """Aggregate encrypted gradients from remote peers."""

    def __init__(self, key: int = 0) -> None:
        self.key = key

    def encrypt(self, grad: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(self.key)
        noise = torch.randn_like(grad)
        self._last_noise = noise
        return grad + noise

    def decrypt(self, grad: torch.Tensor) -> torch.Tensor:
        return grad - getattr(self, "_last_noise", torch.zeros_like(grad))

    def aggregate(self, grads: Iterable[torch.Tensor]) -> torch.Tensor:
        grads = list(grads)
        if not grads:
            raise ValueError("no gradients")
        agg = torch.stack(grads).mean(dim=0)
        return agg

__all__ = ["SecureFederatedLearner"]
