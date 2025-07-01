import unittest
import importlib.machinery
import importlib.util
import sys
import numpy as np
import torch

loader = importlib.machinery.SourceFileLoader('mmw', 'src/multimodal_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mmw = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mmw
sys.modules['asi.multimodal_world_model'] = mmw
sys.modules['multimodal_world_model'] = mmw
loader.exec_module(mmw)
MultiModalWorldModelConfig = mmw.MultiModalWorldModelConfig
MultiModalWorldModel = mmw.MultiModalWorldModel

loader = importlib.machinery.SourceFileLoader('gda', 'src/generative_data_augmentor.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
gda = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = gda
sys.modules['asi.generative_data_augmentor'] = gda
sys.modules['generative_data_augmentor'] = gda
loader.exec_module(gda)
GenerativeDataAugmentor = gda.GenerativeDataAugmentor

loader = importlib.machinery.SourceFileLoader('di', 'src/data_ingest.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
di = importlib.util.module_from_spec(spec)
sys.modules['asi.data_ingest'] = di
loader.exec_module(di)
synthesize_from_world_model = di.synthesize_from_world_model


class TestGenerativeDataAugmentor(unittest.TestCase):
    def setUp(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=2)
        self.world_model = MultiModalWorldModel(cfg)
        self.augmentor = GenerativeDataAugmentor(self.world_model)

    def test_synthesize(self):
        policy = lambda s: torch.zeros(1, dtype=torch.long)
        img = np.zeros((3, 4, 4), dtype=np.float32)
        triples = self.augmentor.synthesize('hi', img, policy, steps=2)
        self.assertEqual(len(triples), 2)
        self.assertEqual(triples[0][1].shape, img.shape)

    def test_pipeline(self):
        policy = lambda s: torch.zeros(1, dtype=torch.long)
        img = np.zeros((3, 4, 4), dtype=np.float32)
        out = synthesize_from_world_model(self.augmentor, [('hello', img)], policy, steps=1)
        self.assertEqual(len(out), 1)


if __name__ == '__main__':
    unittest.main()
