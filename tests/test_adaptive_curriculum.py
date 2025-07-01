import importlib.machinery
import importlib.util
import sys
import unittest
import torch
import types

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

loader_env = importlib.machinery.SourceFileLoader('asi.self_play_env', 'src/self_play_env.py')
spec_env = importlib.util.spec_from_loader(loader_env.name, loader_env)
self_play_env = importlib.util.module_from_spec(spec_env)
sys.modules['asi.self_play_env'] = self_play_env
loader_env.exec_module(self_play_env)
PrioritizedReplayBuffer = self_play_env.PrioritizedReplayBuffer

loader_rst = importlib.machinery.SourceFileLoader('asi.robot_skill_transfer', 'src/robot_skill_transfer.py')
spec_rst = importlib.util.spec_from_loader(loader_rst.name, loader_rst)
robot_skill_transfer = importlib.util.module_from_spec(spec_rst)
sys.modules['asi.robot_skill_transfer'] = robot_skill_transfer
loader_rst.exec_module(robot_skill_transfer)
VideoPolicyDataset = robot_skill_transfer.VideoPolicyDataset

loader_ac = importlib.machinery.SourceFileLoader('asi.adaptive_curriculum', 'src/adaptive_curriculum.py')
spec_ac = importlib.util.spec_from_loader(loader_ac.name, loader_ac)
adaptive_curriculum = importlib.util.module_from_spec(spec_ac)
sys.modules['asi.adaptive_curriculum'] = adaptive_curriculum
loader_ac.exec_module(adaptive_curriculum)
AdaptiveCurriculum = adaptive_curriculum.AdaptiveCurriculum


class TestAdaptiveCurriculum(unittest.TestCase):
    def test_probs_update(self):
        frames = [torch.randn(3, 4, 4) for _ in range(3)]
        actions = [0, 1, 0]
        curated = VideoPolicyDataset(frames, actions)
        buf = PrioritizedReplayBuffer(3)
        for f, a in zip(frames, actions):
            buf.add(f, a, 1.0)
        ac = AdaptiveCurriculum(curated, buf, adaptive_curriculum.CurriculumConfig(lr=0.5))
        p_before = ac._probs()[0].item()
        ac.update(0, 1.0)
        ac.update(1, -1.0)
        p_after = ac._probs()[0].item()
        self.assertGreater(p_after, p_before)


if __name__ == '__main__':
    unittest.main()
