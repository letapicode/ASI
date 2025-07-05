import os
import tempfile
import unittest
from pathlib import Path
import json
import importlib.machinery
import importlib.util
import sys
import types
loader_ana = importlib.machinery.SourceFileLoader('src.dataset_anonymizer', 'src/dataset_anonymizer.py')
spec_ana = importlib.util.spec_from_loader(loader_ana.name, loader_ana)
ana_mod = importlib.util.module_from_spec(spec_ana)
ana_mod.__package__ = 'src'
sys.modules['src.dataset_anonymizer'] = ana_mod
loader_ana.exec_module(ana_mod)
torch = types.SimpleNamespace(
    tensor=lambda *a, **kw: None,
    zeros=lambda *a, **kw: None,
    long=lambda: None,
)
sys.modules['torch'] = torch

loader = importlib.machinery.SourceFileLoader('src.data_ingest', 'src/data_ingest.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
di = importlib.util.module_from_spec(spec)
di.__package__ = 'src'
sys.modules['src.data_ingest'] = di
sys.modules['asi.data_ingest'] = di
loader.exec_module(di)

loader_dlm = importlib.machinery.SourceFileLoader('src.dataset_lineage_manager', 'src/dataset_lineage_manager.py')
spec_dlm = importlib.util.spec_from_loader(loader_dlm.name, loader_dlm)
dlm = importlib.util.module_from_spec(spec_dlm)
dlm.__package__ = 'src'
src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src.dataset_lineage_manager'] = dlm
sys.modules['asi.dataset_lineage_manager'] = dlm
loader_dlm.exec_module(dlm)

pair_modalities = di.pair_modalities
random_crop_image = di.random_crop_image
add_gaussian_noise = di.add_gaussian_noise
text_dropout = di.text_dropout
offline_synthesizer = di.offline_synthesizer
filter_dataset = di.filter_dataset
DatasetVersioner = di.DatasetVersioner
import numpy as np
import asyncio
from unittest.mock import patch

mm = types.SimpleNamespace(
    MultiModalWorldModel=object,
    MultiModalWorldModelConfig=object,
)


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

    def test_download_triples_version(self):
        async def fake_download(session, url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(url)

        with tempfile.TemporaryDirectory() as root:
            urls = ['u1']
            ver = DatasetVersioner(root)
            with patch.object(di, '_download_file_async', fake_download):
                di.download_triples(urls, urls, urls, root, versioner=ver)
            vf = Path(root) / 'dataset_version.json'
            self.assertTrue(vf.exists())
            data = json.loads(vf.read_text())
            self.assertEqual(len(data['files']), 3)

    def test_download_triples_anonymizer(self):
        async def fake_download(session, url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text('call 123-456-7890 or mail foo@bar.com')

        with tempfile.TemporaryDirectory() as root:
            urls = ['u']
            ana = di.DatasetAnonymizer()
            lin = dlm.DatasetLineageManager(root)
            with patch.object(di, '_download_file_async', fake_download):
                di._HAS_AIOHTTP = True
                class DummySession:
                    async def __aenter__(self):
                        return self
                    async def __aexit__(self, exc_type, exc, tb):
                        pass
                di.aiohttp = types.SimpleNamespace(ClientSession=DummySession)
                di.download_triples(urls, urls, urls, root, anonymizer=ana, lineage=lin)
            t = Path(root) / 'text/0.txt'
            self.assertNotIn('@', t.read_text())
            log = json.loads((Path(root) / 'dataset_lineage.json').read_text())
            self.assertIn('anonymized', log[-1]['note'])

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

    def test_offline_synthesizer_version(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2, embed_dim=8)
        wm = MultiModalWorldModel(cfg)

        def policy(state):
            return torch.zeros((), dtype=torch.long)

        def tokenizer(t: str):
            return [ord(c) % cfg.vocab_size for c in t]

        with tempfile.TemporaryDirectory() as root:
            ver = DatasetVersioner(root)
            di.offline_synthesizer(
                wm, tokenizer, 'hi', np.zeros((1, 4, 4), dtype=np.float32), policy,
                steps=1, save_dir=root, versioner=ver
            )
            vf = Path(root) / 'dataset_version.json'
            self.assertTrue(vf.exists())
            data = json.loads(vf.read_text())
            self.assertEqual(len(data['files']), 3)

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

    def test_paraphrase_multilingual(self):
        with tempfile.TemporaryDirectory() as root:
            src = Path(root) / "base.txt"
            src.write_text("hello world")
            trans = di.CrossLingualTranslator(["es", "fr"])
            insp = types.SimpleNamespace(inspect=lambda p: True)
            lin = dlm.DatasetLineageManager(root)
            out = di.paraphrase_multilingual([src], trans, None, insp, lin)
            self.assertTrue(out)
            log = json.loads((Path(root) / "dataset_lineage.json").read_text())
            self.assertIn("paraphrase_multilingual", log[-1]["note"])


if __name__ == '__main__':
    unittest.main()
