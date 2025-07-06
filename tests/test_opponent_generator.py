import unittest
import importlib.machinery
import importlib.util
import sys
import types
import torch
from unittest.mock import patch

pkg = types.ModuleType("asi")
sys.modules["asi"] = pkg

loader_gen = importlib.machinery.SourceFileLoader(
    "asi.opponent_generator", "src/opponent_generator.py"
)
spec_gen = importlib.util.spec_from_loader(loader_gen.name, loader_gen)
opponent_generator = importlib.util.module_from_spec(spec_gen)
sys.modules["asi.opponent_generator"] = opponent_generator
loader_gen.exec_module(opponent_generator)

loader_env = importlib.machinery.SourceFileLoader(
    "asi.self_play_env", "src/self_play_env.py"
)
spec_env = importlib.util.spec_from_loader(loader_env.name, loader_env)
self_play_env = importlib.util.module_from_spec(spec_env)
sys.modules["asi.self_play_env"] = self_play_env
loader_env.exec_module(self_play_env)

loader_rst = importlib.machinery.SourceFileLoader(
    "asi.robot_skill_transfer", "src/robot_skill_transfer.py"
)
spec_rst = importlib.util.spec_from_loader(loader_rst.name, loader_rst)
robot_skill_transfer = importlib.util.module_from_spec(spec_rst)
sys.modules["asi.robot_skill_transfer"] = robot_skill_transfer
loader_rst.exec_module(robot_skill_transfer)

loader_loop = importlib.machinery.SourceFileLoader(
    "asi.self_play_skill_loop", "src/self_play_skill_loop.py"
)
spec_loop = importlib.util.spec_from_loader(loader_loop.name, loader_loop)
self_play_skill_loop = importlib.util.module_from_spec(spec_loop)
sys.modules["asi.self_play_skill_loop"] = self_play_skill_loop
loader_loop.exec_module(self_play_skill_loop)

OpponentGenerator = opponent_generator.OpponentGenerator
run_loop = self_play_skill_loop.run_loop
SelfPlaySkillLoopConfig = self_play_skill_loop.SelfPlaySkillLoopConfig


class TestOpponentGenerator(unittest.TestCase):
    def test_diversifies_over_time(self):
        cfg = SelfPlaySkillLoopConfig(cycles=2, steps=1, epochs=1, batch_size=1)
        frames = [torch.zeros(cfg.img_channels, 4, 4)]
        actions = [0]
        gen = OpponentGenerator()

        def fake_rollout(env, policy, steps=1, return_actions=False):
            obs = [torch.zeros(env.state.shape) for _ in range(steps)]
            rewards = [1.0]
            acts = [0]
            if return_actions:
                return obs, rewards, acts
            return obs, rewards

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0), cfg.action_dim)

        def fake_transfer(c, d):
            return DummyModel()

        policy = lambda obs: torch.zeros_like(obs)
        with (
            patch.object(self_play_skill_loop, "rollout_env", fake_rollout),
            patch.object(self_play_skill_loop, "transfer_skills", fake_transfer),
        ):
            run_loop(cfg, policy, frames, actions, opponent_gen=gen)

        self.assertGreaterEqual(len(gen.policies), 3)
        samples = {gen.sample() for _ in range(5)}
        self.assertGreaterEqual(len(samples), 2)


if __name__ == "__main__":
    unittest.main()
