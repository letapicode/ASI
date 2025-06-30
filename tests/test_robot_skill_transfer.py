import unittest
import torch
from torch import nn

from asi.robot_skill_transfer import transfer_skills


class DummyPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Linear(obs_dim, action_dim)

    def forward(self, obs):
        return self.net(obs)


class TestRobotSkillTransfer(unittest.TestCase):
    def test_transfer(self):
        policy = DummyPolicy(4, 2)
        demos = [
            (torch.randn(1, 4), torch.randn(1, 2))
            for _ in range(3)
        ]
        transfer_skills(policy, demos, epochs=1)


if __name__ == "__main__":
    unittest.main()
