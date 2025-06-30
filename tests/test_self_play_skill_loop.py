import os
import importlib.util
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location("self_play_skill_loop", os.path.join(os.path.dirname(__file__), "..", "src", "self_play_skill_loop.py"))
self_play_skill_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(self_play_skill_loop)
run_loop = self_play_skill_loop.run_loop
SelfPlaySkillLoopConfig = self_play_skill_loop.SelfPlaySkillLoopConfig


class TestSelfPlaySkillLoop(unittest.TestCase):
    def test_run_loop_mocked(self):
        cfg = SelfPlaySkillLoopConfig(cycles=2, steps=3, epochs=1)
        frames = [torch.randn(cfg.img_channels, 4, 4) for _ in range(2)]
        actions = [0, 1]

        def fake_rollout(env, policy, steps=3):
            obs = [torch.zeros(env.state.shape) for _ in range(steps)]
            rewards = [1.0] * steps
            return obs, rewards

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0), cfg.action_dim)

        def fake_transfer(c, d):
            return DummyModel()

        policy = lambda obs: torch.zeros_like(obs)
        with patch.object(self_play_skill_loop, "rollout_env", fake_rollout), patch.object(self_play_skill_loop, "transfer_skills", fake_transfer):
            rewards, model = run_loop(cfg, policy, frames, actions)
        self.assertEqual(len(rewards), 2)
        self.assertIsInstance(model, DummyModel)
        self.assertTrue(all(r == 1.0 for r in rewards))


if __name__ == "__main__":
    unittest.main()
