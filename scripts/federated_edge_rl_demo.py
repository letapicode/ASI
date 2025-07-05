#!/usr/bin/env python
"""Demo of federated Edge RL training."""

from __future__ import annotations

import torch

from asi.federated_edge_rl import FederatedEdgeRLTrainer, FederatedEdgeConfig
from asi.compute_budget_tracker import ComputeBudgetTracker
from asi.differential_privacy_optimizer import DifferentialPrivacyConfig


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def main() -> None:
    data1 = [(torch.randn(1, 2), torch.randn(1, 2)) for _ in range(4)]
    data2 = [(torch.randn(1, 2), torch.randn(1, 2)) for _ in range(4)]

    model = ToyModel()
    cfg = FederatedEdgeConfig(rounds=2, local_steps=2, lr=0.1)
    dp_cfg = DifferentialPrivacyConfig(lr=cfg.lr, clip_norm=1.0, noise_std=0.01)
    budget = ComputeBudgetTracker(1.0)
    budget.start("edge")
    trainer = FederatedEdgeRLTrainer(model, cfg, dp_cfg=dp_cfg, budget=budget)
    trainer.train([data1, data2])
    budget.stop()
    print("trained weights", [p.abs().sum().item() for p in model.parameters()])


if __name__ == "__main__":  # pragma: no cover - demo
    main()
