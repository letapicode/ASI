import unittest
import importlib.machinery
import importlib.util
import types
import sys
import numpy as np
import torch

# load modules dynamically to avoid package deps
loader = importlib.machinery.SourceFileLoader('wmrl', 'src/world_model_rl.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
wmrl = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = wmrl
loader.exec_module(wmrl)
RLBridgeConfig = wmrl.RLBridgeConfig
TransitionDataset = wmrl.TransitionDataset
train_world_model = wmrl.train_world_model

loader = importlib.machinery.SourceFileLoader('gda', 'src/generative_data_augmentor.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
gda = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = gda
loader.exec_module(gda)
GenerativeDataAugmentor = gda.GenerativeDataAugmentor

loader = importlib.machinery.SourceFileLoader('mmw', 'src/multimodal_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mmw = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mmw
loader.exec_module(mmw)
MultiModalWorldModelConfig = mmw.MultiModalWorldModelConfig
MultiModalWorldModel = mmw.MultiModalWorldModel


class TestWorldModel3DTrain(unittest.TestCase):
    def test_train_world_model_with_3d(self):
        cfg = RLBridgeConfig(state_dim=8, action_dim=1, epochs=1, batch_size=1)
        dataset = TransitionDataset([(torch.zeros(8), 0, torch.zeros(8), 0.0)])
        mm_cfg = MultiModalWorldModelConfig(vocab_size=5, img_channels=1, action_dim=1)
        mm = MultiModalWorldModel(mm_cfg)
        augmentor = GenerativeDataAugmentor(mm)
        vol = np.zeros((1, 2, 2, 2), dtype=np.float32)
        policy = lambda s: torch.zeros(1, dtype=torch.long)
        synth = augmentor.synthesize_3d('x', vol, policy, steps=1)
        model = train_world_model(cfg, dataset, synth_3d=synth)
        self.assertIsInstance(model, torch.nn.Module)


if __name__ == '__main__':
    unittest.main()
