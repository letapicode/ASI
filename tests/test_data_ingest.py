import os
import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import sys

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
filter_dataset = di.filter_dataset
import numpy as np


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

    def test_filter_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for i in range(3):
                p = Path(root) / f"g{i}.txt"
                p.write_text("hello world")
                paths.append(p)
            noise = Path(root) / "noise.txt"
            noise.write_text("asdf qwer zxcv")
            paths.append(noise)
            kept = filter_dataset(paths, threshold=-2.5)
            self.assertIn(paths[0], kept)
            self.assertNotIn(noise, kept)


if __name__ == '__main__':
    unittest.main()
