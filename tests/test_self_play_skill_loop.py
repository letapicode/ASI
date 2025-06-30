import unittest
import torch

from asi.self_play_skill_loop import self_play_skill_loop
from asi.self_play_env import SimpleEnv
from asi.robot_skill_transfer import SkillTransferConfig, VideoPolicyDataset


class TestSelfPlaySkillLoop(unittest.TestCase):
    def test_one_cycle(self):
        env = SimpleEnv(state_dim=2)

        def policy(obs: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(obs) * 0.05

        cfg = SkillTransferConfig(img_channels=1, action_dim=2, epochs=1, batch_size=2)
        frames = [torch.randn(1, 4, 4) for _ in range(4)]
        actions = [0, 1, 0, 1]
        dataset = VideoPolicyDataset(frames, actions)

        rewards = self_play_skill_loop(env, policy, cfg, dataset, cycles=1, steps=5)
        self.assertEqual(len(rewards), 1)
        self.assertGreater(len(rewards[0]), 0)


if __name__ == '__main__':
    unittest.main()
