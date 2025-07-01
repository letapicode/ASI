import unittest
from unittest.mock import patch
import importlib.machinery
import importlib.util
import sys
import torch

loader = importlib.machinery.SourceFileLoader('mmwm', 'src/multimodal_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mmwm = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mmwm
sys.modules['asi.multimodal_world_model'] = mmwm
loader.exec_module(mmwm)
MultiModalWorldModelConfig = mmwm.MultiModalWorldModelConfig
MultiModalWorldModel = mmwm.MultiModalWorldModel


class TestCheckpointedWorldModel(unittest.TestCase):
    def setUp(self):
        self.text = torch.randint(0, 10, (1, 4))
        self.img = torch.randn(1, 3, 8, 8)
        self.action = torch.randint(0, 4, (1,))

    def test_no_checkpoint_default(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=4)
        model = MultiModalWorldModel(cfg)
        with patch.object(mmwm, 'checkpoint', side_effect=RuntimeError('called')) as cp, \
             patch('torch.cuda.memory_allocated', side_effect=[100, 100]):
            before = torch.cuda.memory_allocated()
            model(self.text, self.img, self.action)
            after = torch.cuda.memory_allocated()
            self.assertFalse(cp.called)
            self.assertEqual(before, after)

    def test_checkpoint_reduces_memory(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=4, checkpoint_blocks=True)
        model = MultiModalWorldModel(cfg)

        def passthrough(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(mmwm, 'checkpoint', side_effect=passthrough) as cp, \
             patch('torch.cuda.memory_allocated', side_effect=[100, 80]):
            before = torch.cuda.memory_allocated()
            model(self.text, self.img, self.action)
            after = torch.cuda.memory_allocated()
            self.assertTrue(cp.called)
            self.assertLess(after, before)


if __name__ == '__main__':
    unittest.main()
