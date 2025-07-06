from __future__ import annotations

import torch
from typing import Iterable, Set

from .dataset_discovery import DiscoveredDataset


class DatasetQualityAgent:
    """Simple RL agent to score discovered datasets."""

    def __init__(self, allowed_licenses: Iterable[str] | None = None, lr: float = 0.1) -> None:
        self.allowed = {l.lower() for l in (allowed_licenses or ["mit", "apache", "cc-by"])}
        self.lr = lr
        self.weights = torch.zeros(3, dtype=torch.float32)
        self.baseline = 0.0
        self.seen: Set[str] = set()

    # --------------------------------------------------------------
    def _signals(self, d: DiscoveredDataset) -> torch.Tensor:
        lic_text = (d.license + " " + d.license_text).lower()
        license_ok = 1.0 if any(a in lic_text for a in self.allowed) else 0.0
        tokens = d.name.split()
        diversity = len(set(tokens)) / max(len(tokens), 1)
        novelty = 0.0 if d.name in self.seen else 1.0
        return torch.tensor([license_ok, diversity, novelty], dtype=torch.float32)

    # --------------------------------------------------------------
    def evaluate(self, d: DiscoveredDataset) -> float:
        """Return weight for ``d`` and update the agent."""
        x = self._signals(d)
        prob = torch.sigmoid((self.weights * x).sum())
        reward = x.mean()
        self.baseline += self.lr * (reward - self.baseline)
        adv = reward - self.baseline
        self.weights += self.lr * adv * x
        self.seen.add(d.name)
        return float(prob.item())


__all__ = ["DatasetQualityAgent"]
