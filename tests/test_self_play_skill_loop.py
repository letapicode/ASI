import unittest
from unittest.mock import patch
import torch

import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
loader_env = importlib.machinery.SourceFileLoader('asi.self_play_env', 'src/self_play_env.py')
spec_env = importlib.util.spec_from_loader(loader_env.name, loader_env)
self_play_env = importlib.util.module_from_spec(spec_env)
sys.modules['asi.self_play_env'] = self_play_env
loader_env.exec_module(self_play_env)
loader_rst = importlib.machinery.SourceFileLoader('asi.robot_skill_transfer', 'src/robot_skill_transfer.py')
spec_rst = importlib.util.spec_from_loader(loader_rst.name, loader_rst)
robot_skill_transfer = importlib.util.module_from_spec(spec_rst)
sys.modules['asi.robot_skill_transfer'] = robot_skill_transfer
loader_rst.exec_module(robot_skill_transfer)
loader_ac = importlib.machinery.SourceFileLoader('asi.adaptive_curriculum', 'src/adaptive_curriculum.py')
spec_ac = importlib.util.spec_from_loader(loader_ac.name, loader_ac)
adaptive_curriculum = importlib.util.module_from_spec(spec_ac)
sys.modules['asi.adaptive_curriculum'] = adaptive_curriculum
loader_ac.exec_module(adaptive_curriculum)
loader = importlib.machinery.SourceFileLoader('asi.self_play_skill_loop', 'src/self_play_skill_loop.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
self_play_skill_loop = importlib.util.module_from_spec(spec)
sys.modules['asi.self_play_skill_loop'] = self_play_skill_loop
loader.exec_module(self_play_skill_loop)
run_loop = self_play_skill_loop.run_loop
SelfPlaySkillLoopConfig = self_play_skill_loop.SelfPlaySkillLoopConfig
AdaptiveCurriculum = self_play_skill_loop.AdaptiveCurriculum


class TestSelfPlaySkillLoop(unittest.TestCase):
    def test_run_loop_mocked(self):
        cfg = SelfPlaySkillLoopConfig(cycles=2, steps=3, epochs=1)
        frames = [torch.randn(cfg.img_channels, 4, 4) for _ in range(2)]
        actions = [0, 1]

        def fake_rollout(env, policy, steps=3, return_actions=False):
            obs = [torch.zeros(env.state.shape) for _ in range(steps)]
            rewards = [1.0] * steps
            actions = [0 for _ in range(steps)]
            if return_actions:
                return obs, rewards, actions
            return obs, rewards

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0), cfg.action_dim)

        def fake_transfer(c, d):
            return DummyModel()

        policy = lambda obs: torch.zeros_like(obs)

        class DummyCurriculum:
            def __init__(self, *a, **kw):
                pass

            def sample(self, bs):
                return frames[:bs], actions[:bs], 0

            def update(self, idx, reward):
                pass

        with patch.object(self_play_skill_loop, "rollout_env", fake_rollout), \
             patch.object(self_play_skill_loop, "transfer_skills", fake_transfer), \
             patch.object(self_play_skill_loop, "AdaptiveCurriculum", DummyCurriculum):
            rewards, model = run_loop(cfg, policy, frames, actions)
        self.assertEqual(len(rewards), 2)
        self.assertIsInstance(model, DummyModel)
        self.assertTrue(all(r == 1.0 for r in rewards))


if __name__ == '__main__':
    unittest.main()
