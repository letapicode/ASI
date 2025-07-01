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
import numpy as np
import asyncio
from unittest.mock import patch


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

    def test_download_triples(self):
        async def fake_download(session, url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(url)

        with tempfile.TemporaryDirectory() as root:
            urls = ['u1']
            with patch.object(di, '_download_file_async', fake_download):
                triples = di.download_triples(urls, urls, urls, root)
            self.assertEqual(len(triples), 1)
            t, i, a = triples[0]
            self.assertTrue(t.exists() and i.exists() and a.exists())
            self.assertEqual(t.read_text(), 'u1')

    def test_download_triples_event_loop(self):
        async def fake_download(session, url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(url)

        async def run():
            with tempfile.TemporaryDirectory() as root:
                urls = ['u']
                with patch.object(di, '_download_file_async', fake_download):
                    task = di.download_triples(urls, urls, urls, root)
                    self.assertTrue(asyncio.isfuture(task))
                    triples = await task
                    self.assertEqual(len(triples), 1)

        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
