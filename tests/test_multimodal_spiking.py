import unittest
import importlib.machinery
import importlib.util
import sys
import types
import torch

loader = importlib.machinery.SourceFileLoader('mmwm', 'src/multimodal_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mmwm = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mmwm
mmwm.__package__ = 'asi'
sys.modules['asi.multimodal_world_model'] = mmwm
pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
loader_lb = importlib.machinery.SourceFileLoader('asi.loihi_backend', 'src/loihi_backend.py')
spec_lb = importlib.util.spec_from_loader(loader_lb.name, loader_lb)
lb = importlib.util.module_from_spec(spec_lb)
lb.__package__ = 'asi'
sys.modules['asi.loihi_backend'] = lb
loader_lb.exec_module(lb)
loader_sl = importlib.machinery.SourceFileLoader('asi.spiking_layers', 'src/spiking_layers.py')
spec_sl = importlib.util.spec_from_loader(loader_sl.name, loader_sl)
sl = importlib.util.module_from_spec(spec_sl)
sl.__package__ = 'asi'
sys.modules['asi.spiking_layers'] = sl
loader_sl.exec_module(sl)
loader.exec_module(mmwm)

MultiModalWorldModelConfig = mmwm.MultiModalWorldModelConfig
MultiModalWorldModel = mmwm.MultiModalWorldModel

class TestSpikingWorldModel(unittest.TestCase):
    def setUp(self):
        self.text = torch.randint(0, 10, (1, 4))
        self.img = torch.randn(1, 3, 8, 8)
        self.action = torch.randint(0, 4, (1,))

    def test_spiking_output_close(self):
        cfg_dense = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=4)
        cfg_spike = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=4, use_spiking=True)
        dense = MultiModalWorldModel(cfg_dense)
        spiking = MultiModalWorldModel(cfg_spike)
        out_dense = dense(self.text, self.img, self.action)
        out_spike = spiking(self.text, self.img, self.action)
        self.assertEqual(out_dense[0].shape, out_spike[0].shape)
        self.assertEqual(out_dense[1].shape, out_spike[1].shape)
        diff = torch.mean(torch.abs(out_dense[0] - out_spike[0])).item()
        self.assertLess(diff, 1.0)


if __name__ == '__main__':
    unittest.main()
