from __future__ import annotations
from typing import Iterable, List, Tuple

import torch

from .compute_budget_tracker import ComputeBudgetTracker
from .adaptive_micro_batcher import AdaptiveMicroBatcher


class EdgeRLTrainer:
    """Run world-model updates with compute budget checks."""

    def __init__(
        self,
        model,
        optimizer,
        budget: ComputeBudgetTracker,
        run_id: str = "edge",
        micro_batcher: AdaptiveMicroBatcher | None = None,
    ) -> None:
        self.model = model
        self.opt = optimizer
        self.budget = budget
        self.run_id = run_id
        self.micro_batcher = micro_batcher

    def train(
        self,
        data: Iterable[tuple[torch.Tensor, torch.Tensor]],
        threshold: float = 0.1,
    ) -> int:
        steps = 0
        micro = self.micro_batcher
        batches: Iterable[List[Tuple[torch.Tensor, torch.Tensor]]]
        if micro is not None:
            micro.start()
            batches = micro.micro_batches(data)
        else:
            batches = [[p] for p in data]

        for batch in batches:
            if self.budget.remaining(self.run_id) <= threshold:
                break
            states = torch.cat([s.unsqueeze(0) for s, _ in batch], dim=0)
            targets = torch.cat([t.unsqueeze(0) for _, t in batch], dim=0)
            pred = self.model(states)
            loss = torch.nn.functional.mse_loss(pred, targets)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            steps += 1
            if micro is not None:
                micro.tick()

        if micro is not None:
            micro.stop()
        return steps

__all__ = ["EdgeRLTrainer"]
