import unittest
import torch
from torch import nn

from asi.world_model_rl import WorldModel, rollout_model


class DummyPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, action_dim)

    def forward(self, x):
        return self.fc(x)


class TestWorldModelRL(unittest.TestCase):
    def test_rollout_model(self):
        model = WorldModel(obs_dim=4, action_dim=3)
        policy = DummyPolicy(4, 3)
        obs = torch.zeros(1, 4)
        actions = torch.zeros(1, 1, dtype=torch.long)
        next_obs = torch.zeros(1, 1, 4)
        reward = torch.zeros(1, 1)
        dataset = [(obs, actions, next_obs, reward)]
        rollout_model(model, obs.squeeze(0), policy, steps=1)
        train_world_model = __import__('asi.world_model_rl').world_model_rl.train_world_model
        train_world_model(model, dataset, epochs=1)


if __name__ == '__main__':
    unittest.main()
