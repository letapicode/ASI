import unittest
import importlib.machinery
import importlib.util
import sys
import numpy as np
import torch

# load diffusion_world_model
loader = importlib.machinery.SourceFileLoader('dwm', 'src/diffusion_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
dwm = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = dwm
sys.modules['asi.diffusion_world_model'] = dwm
loader.exec_module(dwm)
DiffusionWorldModel = dwm.DiffusionWorldModel

# load generative_data_augmentor and multimodal_world_model
loader = importlib.machinery.SourceFileLoader('mmw', 'src/multimodal_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mmw = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mmw
sys.modules['multimodal_world_model'] = mmw
loader.exec_module(mmw)
MultiModalWorldModelConfig = mmw.MultiModalWorldModelConfig
MultiModalWorldModel = mmw.MultiModalWorldModel

loader = importlib.machinery.SourceFileLoader('gda', 'src/generative_data_augmentor.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
gda = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = gda
sys.modules['generative_data_augmentor'] = gda
loader.exec_module(gda)
GenerativeDataAugmentor = gda.GenerativeDataAugmentor


class TestDiffusionWorldModel(unittest.TestCase):
    def test_sample(self):
        model = DiffusionWorldModel(state_dim=4)
        init = torch.zeros(4)
        states = model.sample(init, steps=2)
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0].shape, init.shape)

    def test_augmentor_integration(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=2)
        wm = MultiModalWorldModel(cfg)
        diff = DiffusionWorldModel(state_dim=cfg.embed_dim)
        augmentor = GenerativeDataAugmentor(wm, diffusion_model=diff)
        policy = lambda s: torch.zeros(1, dtype=torch.long)
        img = np.zeros((3, 4, 4), dtype=np.float32)
        triples = augmentor.synthesize('hi', img, policy, steps=1)
        self.assertGreater(len(triples), 1)


if __name__ == '__main__':
    unittest.main()
