import os
import tempfile
import unittest
from pathlib import Path
import json
import importlib.machinery
import importlib.util
import sys
import types
import wave
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

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg

loader = importlib.machinery.SourceFileLoader('src.data_ingest', 'src/data_ingest.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
di = importlib.util.module_from_spec(spec)
di.__package__ = 'src'
sys.modules['src.data_ingest'] = di
sys.modules['asi.data_ingest'] = di
loader.exec_module(di)

loader_dlm = importlib.machinery.SourceFileLoader('src.dataset_lineage', 'src/dataset_lineage.py')
spec_dlm = importlib.util.spec_from_loader(loader_dlm.name, loader_dlm)
dlm = importlib.util.module_from_spec(spec_dlm)
dlm.__package__ = 'src'
src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src.dataset_lineage'] = dlm
loader_poison = importlib.machinery.SourceFileLoader('src.data_poison_detector', 'src/data_poison_detector.py')
spec_p = importlib.util.spec_from_loader(loader_poison.name, loader_poison)
poison_mod = importlib.util.module_from_spec(spec_p)
poison_mod.__package__ = 'src'
sys.modules['src.data_poison_detector'] = poison_mod
loader_poison.exec_module(poison_mod)
sys.modules['asi.data_poison_detector'] = poison_mod
sys.modules['asi.dataset_lineage'] = dlm
loader_dlm.exec_module(dlm)
DataPoisonDetector = poison_mod.DataPoisonDetector

loader_priv = importlib.machinery.SourceFileLoader('src.privacy', 'src/privacy.py')
spec_priv = importlib.util.spec_from_loader(loader_priv.name, loader_priv)
priv_mod = importlib.util.module_from_spec(spec_priv)
priv_mod.__package__ = 'src'
sys.modules['src.privacy'] = priv_mod
sys.modules['asi.privacy'] = priv_mod
loader_priv.exec_module(priv_mod)

loader_li = importlib.machinery.SourceFileLoader('src.license_inspector', 'src/license_inspector.py')
spec_li = importlib.util.spec_from_loader(loader_li.name, loader_li)
li_mod = importlib.util.module_from_spec(spec_li)
li_mod.__package__ = 'src'
sys.modules['src.license_inspector'] = li_mod
loader_li.exec_module(li_mod)

loader_wm = importlib.machinery.SourceFileLoader('src.dataset_watermarker', 'src/dataset_watermarker.py')
spec_wm = importlib.util.spec_from_loader(loader_wm.name, loader_wm)
wm_mod = importlib.util.module_from_spec(spec_wm)
wm_mod.__package__ = 'src'
sys.modules['src.dataset_watermarker'] = wm_mod
loader_wm.exec_module(wm_mod)

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

class DummyWM:
    def __init__(self, *a, **kw):
        pass


class DummyCfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


MultiModalWorldModel = DummyWM
MultiModalWorldModelConfig = DummyCfg


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
        async def fake_download(session, url, dest, watermark_id=None):
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
        async def fake_download(session, url, dest, watermark_id=None):
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

    def test_download_triples_watermark(self):
        async def fake_download(session, url, dest, watermark_id=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(url)
            if watermark_id:
                wm_mod.add_watermark(dest, watermark_id)

        with tempfile.TemporaryDirectory() as root:
            urls = ['u1']
            with patch.object(di, '_download_file_async', fake_download):
                di.download_triples(urls, urls, urls, root, watermark_id='wm1')
            t = Path(root) / 'text/0.txt'
            self.assertEqual(wm_mod.detect_watermark(t), 'wm1')

    def test_download_triples_anonymizer(self):
        async def fake_download(session, url, dest, watermark_id=None):
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
        async def fake_download(session, url, dest, watermark_id=None):
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

    def test_download_triples_poison(self):
        async def fake_download(session, url, dest, watermark_id=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(url)

        with tempfile.TemporaryDirectory() as root:
            urls = [' '.join(f'w{i}' for i in range(20))]
            det = DataPoisonDetector(window=1, threshold=2.0)
            with patch.object(di, '_download_file_async', fake_download):
                triples = di.download_triples(urls, urls, urls, root, poison_detector=det)
            self.assertEqual(len(triples), 0)

    def test_download_triples_audit(self):
        async def fake_download(session, url, dest, watermark_id=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text('x')

        with tempfile.TemporaryDirectory() as root:
            urls = ['u']
            pbm = priv_mod.PrivacyBudgetManager(1.0, 1e-5, Path(root)/'b.json')
            insp = li_mod.LicenseInspector(['mit'])
            lin = dlm.DatasetLineageManager(root)
            auditor = priv_mod.PrivacyAuditor(pbm, insp, lin, report_dir=root)
            meta = Path(root)/'text'/ '0.json'
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps({'license': 'MIT'}))
            with patch.object(di, '_download_file_async', fake_download):
                di._HAS_AIOHTTP = True
                class DummySession:
                    async def __aenter__(self):
                        return self
                    async def __aexit__(self, exc_type, exc, tb):
                        pass
                di.aiohttp = types.SimpleNamespace(ClientSession=DummySession)
                di.download_triples(urls, urls, urls, root, auditor=auditor)
            reports = list(Path(root).glob('download_triples_*.json'))
            self.assertTrue(reports)

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

    def test_ingest_stats(self):
        class DummyDataset(list):
            def __init__(self, items, tok):
                super().__init__(items)
                self.tok = tok

            def __getitem__(self, idx):
                t, i, a = super().__getitem__(idx)
                import torch
                return (
                    torch.tensor(self.tok(t), dtype=torch.long),
                    torch.tensor(i),
                    torch.tensor(a),
                )

            def __len__(self):
                return super().__len__()

        def dummy_encode_all(model, dataset, batch_size=4):
            import torch
            n = len(dataset)
            return torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n, 2)

        with tempfile.TemporaryDirectory() as root:
            t = Path(root) / "0.txt"
            i = Path(root) / "0.npy"
            a = Path(root) / "0.wav"
            t.write_text("hi")
            np.save(i, np.zeros((2, 2), dtype=np.float32))
            with wave.open(str(a), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(np.zeros(2, dtype=np.int16))

            mem = types.SimpleNamespace(add_multimodal=lambda *a, **kw: None)

            tok = lambda x: list(range(len(x)))

            dummy_mod = types.ModuleType("src.cross_modal_fusion")
            dummy_mod.MultiModalDataset = DummyDataset
            dummy_mod.encode_all = dummy_encode_all
            with patch.dict(sys.modules, {"src.cross_modal_fusion": dummy_mod}):
                vecs = di.ingest_translated_triples(
                    [(t, i, a)], tok, object(), mem, batch_size=1, return_stats=True
                )
            self.assertEqual(vecs[3]["text_tokens"], 2)
            self.assertEqual(vecs[3]["image_pixels"], 4)
            self.assertEqual(vecs[3]["audio_samples"], 2)


if __name__ == '__main__':
    unittest.main()
