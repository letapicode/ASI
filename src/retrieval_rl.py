"""Reinforcement-learning policy for retrieval ranking."""

from __future__ import annotations

import random
from typing import Any, Iterable, List


class RetrievalPolicy:
    """Select memory vectors based on learned rewards."""

    def __init__(self, epsilon: float = 0.1, alpha: float = 0.5) -> None:
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.q: dict[Any, float] = {}

    # ------------------------------------------------------
    def rank(self, metas: List[Any], scores: List[float] | None = None) -> List[int]:
        """Return a ranking of indices for ``metas`` by Q-value."""
        if not metas:
            return []
        if random.random() < self.epsilon:
            idx = list(range(len(metas)))
            random.shuffle(idx)
            return idx
        qvals = [self.q.get(m, 0.0) for m in metas]
        return sorted(range(len(metas)), key=lambda i: qvals[i], reverse=True)

    # ------------------------------------------------------
    def update(self, meta: Any, reward: float) -> None:
        """Update Q-value for ``meta`` with observed ``reward``."""
        current = self.q.get(meta, 0.0)
        self.q[meta] = current + self.alpha * (reward - current)


def train_policy(policy: RetrievalPolicy, logs: Iterable[tuple[Any, float]], cycles: int = 1) -> None:
    """Train ``policy`` from ``(meta, reward)`` pairs."""
    entries = list(logs)
    for _ in range(cycles):
        for meta, reward in entries:
            policy.update(meta, float(reward))

