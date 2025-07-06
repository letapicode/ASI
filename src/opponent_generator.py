from __future__ import annotations

from typing import Callable, List, Tuple

import torch


class OpponentGenerator:
    """Generate and evolve opponent policies based on past rewards."""

    def __init__(self) -> None:
        self.policies: List[Tuple[Callable[[torch.Tensor], torch.Tensor], float]] = []

    def update(self, policy: Callable[[torch.Tensor], torch.Tensor], reward: float) -> None:
        """Add a policy with its associated mean reward."""
        self.policies.append((policy, float(reward)))

    def sample(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Sample an opponent weighted by past rewards."""
        if not self.policies:
            raise ValueError("no recorded opponents")
        weights = torch.tensor([max(r, 1e-6) for _, r in self.policies], dtype=torch.float)
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, 1).item()
        return self.policies[idx][0]


__all__ = ["OpponentGenerator"]
