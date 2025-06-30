import unittest
import torch
from asi.world_model_rl import (
    RLBridgeConfig,
    TransitionDataset,
    train_world_model,
    rollout_policy,
)


def _make_dataset(n=6):
    data = []
    for i in range(n):
        state = torch.randn(5)
        action = i % 3
        next_state = state + 0.1
        reward = float(i)
        data.append((state, action, next_state, reward))
    return TransitionDataset(data)


class TestWorldModelRL(unittest.TestCase):
    def test_train_and_rollout(self):
        cfg = RLBridgeConfig(state_dim=5, action_dim=3, hidden_dim=8, lr=0.01, batch_size=2, epochs=1)
        ds = _make_dataset(8)
        model = train_world_model(cfg, ds)
        init = torch.zeros(5)
        policy = lambda s: torch.zeros(s.size(0), dtype=torch.long)
        states, rewards = rollout_policy(model, policy, init, steps=3)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(rewards), 3)


if __name__ == "__main__":
    unittest.main()
