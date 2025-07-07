#!/usr/bin/env python
"""Demo of aggregating EEG rewards across clients."""

import numpy as np
import torch

from asi.world_model_rl import RLBridgeConfig
from asi.bci_feedback_trainer import BCIFeedbackTrainer
from asi.secure_federated_learner import SecureFederatedLearner
from asi.federated_world_model_trainer import (
    FederatedWorldModelTrainer,
    FederatedTrainerConfig,
)


def main() -> None:
    cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=2)
    states = [torch.zeros(2), torch.zeros(2)]
    actions = [0, 1]
    next_states = [torch.ones(2), torch.ones(2)]

    # EEG signals from two clients
    client1 = [np.random.randn(4), np.random.randn(4)]
    client2 = [np.random.randn(4), np.random.randn(4)]

    learner = SecureFederatedLearner()
    bci = BCIFeedbackTrainer(cfg)

    dataset = bci.build_dataset(
        states,
        actions,
        next_states,
        None,
        signals_nodes=[client1, client2],
        learner=learner,
    )

    trainer = FederatedWorldModelTrainer(
        cfg, [dataset], trainer_cfg=FederatedTrainerConfig(rounds=1)
    )
    trainer.train()
    print("trained world model")


if __name__ == "__main__":  # pragma: no cover - example script
    main()
