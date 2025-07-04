from __future__ import annotations
from typing import Iterable

import torch

from .compute_budget_tracker import ComputeBudgetTracker


class EdgeRLTrainer:
    """Run world-model updates with compute budget checks."""

    def __init__(self, model, optimizer, budget: ComputeBudgetTracker, run_id: str = "edge") -> None:
        self.model = model
        self.opt = optimizer
        self.budget = budget
        self.run_id = run_id

    def train(self, data: Iterable[tuple[torch.Tensor, torch.Tensor]], threshold: float = 0.1) -> int:
        steps = 0
        for state, target in data:
            if self.budget.remaining(self.run_id) <= threshold:
                break
            pred = self.model(state)
            loss = torch.nn.functional.mse_loss(pred, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            steps += 1
        return steps

__all__ = ["EdgeRLTrainer"]
