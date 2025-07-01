import os
import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import torch

loader = importlib.machinery.SourceFileLoader('di', 'src/data_ingest.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
di = importlib.util.module_from_spec(spec)
sys.modules['data_ingest'] = di
sys.modules['asi.data_ingest'] = di
loader.exec_module(di)

pair_modalities = di.pair_modalities
random_crop_image = di.random_crop_image
add_gaussian_noise = di.add_gaussian_noise
text_dropout = di.text_dropout
offline_synthesizer = di.offline_synthesizer
import numpy as np

loader_mm = importlib.machinery.SourceFileLoader('mm', 'src/multimodal_world_model.py')
spec_mm = importlib.util.spec_from_loader(loader_mm.name, loader_mm)
mm = importlib.util.module_from_spec(spec_mm)
sys.modules['multimodal_world_model'] = mm
sys.modules['asi.multimodal_world_model'] = mm
sys.modules['mm'] = mm
loader_mm.exec_module(mm)
MultiModalWorldModel = mm.MultiModalWorldModel
MultiModalWorldModelConfig = mm.MultiModalWorldModelConfig


class TestDataIngest(unittest.TestCase):
    def test_pair_and_aug(self):
        with tempfile.TemporaryDirectory() as root:
            td = Path(root) / 'text'
            id = Path(root) / 'img'
            ad = Path(root) / 'aud'
            td.mkdir()
            id.mkdir()
            ad.mkdir()
            (td / 'sample.txt').write_text('hello')
            (id / 'sample.jpg').write_text('img')
            (ad / 'sample.wav').write_text('aud')
            pairs = pair_modalities(td, id, ad)
            self.assertEqual(len(pairs), 1)

        img = np.zeros((4, 4, 1), dtype=np.float32)
        crop = random_crop_image(img, (2, 2))
        self.assertEqual(crop.shape, (2, 2, 1))

        audio = np.zeros(10, dtype=np.float32)
        noisy = add_gaussian_noise(audio, std=0.1)
        self.assertEqual(noisy.shape, audio.shape)

        out = text_dropout('the quick brown fox', p=1.0)
        self.assertTrue(out)

    def test_offline_synthesizer(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2, embed_dim=8)
        wm = MultiModalWorldModel(cfg)

        def policy(state):
            return torch.zeros((), dtype=torch.long)

        def tokenizer(t: str):
            return [ord(c) % cfg.vocab_size for c in t]

        img = np.zeros((1, 4, 4), dtype=np.float32)
        triples = offline_synthesizer(wm, tokenizer, "hi", img, policy, steps=2)

        self.assertEqual(len(triples), 2)
        t, i, a = triples[0]
        self.assertIsInstance(t, str)
        self.assertIsInstance(i, np.ndarray)
        self.assertIsInstance(a, np.ndarray)


if __name__ == '__main__':
    unittest.main()
