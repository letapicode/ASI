import unittest
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
import numpy as np
import torch

loader = importlib.machinery.SourceFileLoader('src.multimodal_world_model', 'src/multimodal_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mmw = importlib.util.module_from_spec(spec)
mmw.__package__ = 'src'
sys.modules['src.multimodal_world_model'] = mmw
loader.exec_module(mmw)
MultiModalWorldModelConfig = mmw.MultiModalWorldModelConfig
MultiModalWorldModel = mmw.MultiModalWorldModel

loader = importlib.machinery.SourceFileLoader('src.generative_data_augmentor', 'src/generative_data_augmentor.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
gda = importlib.util.module_from_spec(spec)
gda.__package__ = 'src'
sys.modules['src.generative_data_augmentor'] = gda
loader.exec_module(gda)
GenerativeDataAugmentor = gda.GenerativeDataAugmentor


class TestGenerative3D(unittest.TestCase):
    def test_synthesize_3d(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2)
        wm = MultiModalWorldModel(cfg)
        augmentor = GenerativeDataAugmentor(wm)
        vol = np.zeros((1, 2, 2, 2), dtype=np.float32)
        policy = lambda s: torch.zeros(1, dtype=torch.long)
        out = augmentor.synthesize_3d('x', vol, policy, steps=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1].shape, vol.shape)


if __name__ == '__main__':
    unittest.main()
