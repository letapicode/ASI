import unittest
from unittest.mock import patch
import importlib.machinery
import importlib.util
import sys
import types
import torch

loader = importlib.machinery.SourceFileLoader('mmwm', 'src/multimodal_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mmwm = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mmwm
pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
loader_sl = importlib.machinery.SourceFileLoader('asi.spiking_layers', 'src/spiking_layers.py')
spec_sl = importlib.util.spec_from_loader(loader_sl.name, loader_sl)
sl = importlib.util.module_from_spec(spec_sl)
sl.__package__ = 'asi'
sys.modules['asi.spiking_layers'] = sl
loader_sl.exec_module(sl)
sys.modules['asi.multimodal_world_model'] = mmwm
mmwm.__package__ = 'asi'
loader_lq = importlib.machinery.SourceFileLoader('asi.lora_quant', 'src/lora_quant.py')
spec_lq = importlib.util.spec_from_loader(loader_lq.name, loader_lq)
lq = importlib.util.module_from_spec(spec_lq)
lq.__package__ = 'asi'
sys.modules['asi.lora_quant'] = lq
loader_lq.exec_module(lq)
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

    def test_lora_forward_runs(self):
        from asi.lora_quant import LoRAQuantLinear

        cfg = MultiModalWorldModelConfig(
            vocab_size=10,
            img_channels=3,
            action_dim=4,
            use_lora=True,
        )
        model = MultiModalWorldModel(cfg)
        self.assertIsInstance(model.dyn.state_proj, LoRAQuantLinear)
        state, reward = model(self.text, self.img, self.action)
        self.assertEqual(state.shape, (1, cfg.embed_dim))
        self.assertEqual(reward.shape, (1,))


if __name__ == '__main__':
    unittest.main()
