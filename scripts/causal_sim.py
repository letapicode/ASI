#!/usr/bin/env python
"""Example demonstrating counterfactual planning."""

from __future__ import annotations

import argparse
import torch

from asi.world_model_rl import (
    RLBridgeConfig,
    TransitionDataset,
    train_world_model,
    simulate_counterfactual,
)
from asi.causal_graph_learner import CausalGraphLearner


def build_dataset() -> TransitionDataset:
    data = []
    for i in range(5):
        s = torch.tensor([float(i), float(i)])
        a = 0
        ns = s + torch.tensor([1.0, 0.5])
        r = float(i)
        data.append((s, a, ns, r))
    return TransitionDataset(data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Counterfactual simulation example")
    parser.parse_args(argv)

    cfg = RLBridgeConfig(state_dim=2, action_dim=1, epochs=1, batch_size=2)
    dataset = build_dataset()
    learner = CausalGraphLearner()
    model = train_world_model(cfg, dataset, learner=learner)

    state = torch.tensor([0.0, 0.0])
    action = torch.tensor(0)
    next_state, _ = simulate_counterfactual(model, learner, state, action, {0: 2.0})
    print("Predicted next state with intervention:", next_state.tolist())
    print("Learned edges:", learner.edges())


if __name__ == "__main__":  # pragma: no cover - example CLI
    main()
