import unittest
import tempfile
import types
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import wave
from PIL import Image

# Set up src package
src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg

# Stub heavy dependencies
torch_stub = types.SimpleNamespace(
    tensor=lambda *a, **kw: None,
    zeros=lambda *a, **kw: None,
    long=lambda: None,
)
sys.modules['torch'] = torch_stub
requests_stub = types.SimpleNamespace(get=lambda *a, **k: types.SimpleNamespace(content=b'', raise_for_status=lambda: None))
sys.modules['requests'] = requests_stub
psutil_stub = types.SimpleNamespace(cpu_percent=lambda interval=None: 0.0)
sys.modules['psutil'] = psutil_stub
sys.modules['pynvml'] = types.ModuleType('pynvml')

# Load anonymizer module
loader = importlib.machinery.SourceFileLoader('src.anonymizer', 'src/anonymizer.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
ana_mod = importlib.util.module_from_spec(spec)
ana_mod.__package__ = 'src'
sys.modules['src.anonymizer'] = ana_mod
loader.exec_module(ana_mod)
DatasetAnonymizer = ana_mod.DatasetAnonymizer
NERAnonymizer = ana_mod.NERAnonymizer

# Load data_ingest for integration tests
loader_di = importlib.machinery.SourceFileLoader('src.data_ingest', 'src/data_ingest.py')
spec_di = importlib.util.spec_from_loader(loader_di.name, loader_di)
di_mod = importlib.util.module_from_spec(spec_di)
di_mod.__package__ = 'src'
sys.modules['src.data_ingest'] = di_mod
loader_di.exec_module(di_mod)


class TestDatasetAnonymizer(unittest.TestCase):
    def test_scrub_text(self):
        da = DatasetAnonymizer()
        out = da.scrub_text("contact me at foo@bar.com or 123-456-7890")
        self.assertNotIn("@", out)
        self.assertNotIn("123-456-7890", out)
        self.assertEqual(da.summary()["text"], 1)

    def test_scrub_files(self):
        da = DatasetAnonymizer()
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp) / "t.txt"
            i = Path(tmp) / "i.png"
            a = Path(tmp) / "a.wav"
            t.write_text("a@b.com")
            img = Image.new("RGB", (2, 2), color=1)
            img.save(i)
            with wave.open(str(a), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(8000)
                f.writeframes(np.ones(8, dtype=np.int16).tobytes())
            da.scrub_text_file(t)
            da.scrub_image_file(i)
            da.scrub_audio_file(a)
            self.assertEqual(t.read_text(), "[EMAIL]")
            self.assertTrue(np.all(np.array(Image.open(i)) == 0))
            with wave.open(str(a)) as f:
                data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
            self.assertTrue(np.all(data == 0))
            summ = da.summary()
            self.assertEqual(summ["text"], 1)
            self.assertEqual(summ["image"], 1)
            self.assertEqual(summ["audio"], 1)


class TestNERAnonymizer(unittest.TestCase):
    def test_scrub_text_and_files(self):
        ner = NERAnonymizer()
        txt = ner.anonymize('Alice works at OpenAI')
        self.assertIn('[PERSON]', txt)
        self.assertIn('[ORG]', txt)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'cap.txt'
            p.write_text('Bob from Google')
            ner.scrub_text_file(p)
            data = p.read_text()
            self.assertIn('[PERSON]', data)
            self.assertIn('[ORG]', data)
        summ = ner.summary()
        self.assertGreaterEqual(summ.get('PERSON', 0), 2)
        self.assertGreaterEqual(summ.get('ORG', 0), 2)

    def test_custom_patterns_and_format(self):
        ner = NERAnonymizer(extra_patterns={"LOC": ["Paris"]}, tag_format="<{label}>")
        txt = ner.anonymize("Trip to Paris")
        self.assertIn("<LOC>", txt)
        self.assertEqual(ner.summary().get("LOC"), 1)

    def test_download_triples_ner(self):
        async def fake_download(session, url, dest, watermark_id=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text('Charlie at NASA')
            if dest.suffix == '.png':
                dest.with_suffix('.caption.txt').write_text('Alice at Google')
            if dest.suffix == '.wav':
                dest.with_suffix('.transcript.txt').write_text('Bob from OpenAI')

        with tempfile.TemporaryDirectory() as root:
            urls = ['u']
            ner = NERAnonymizer()
            with patch.object(di_mod, '_download_file_async', fake_download):
                di_mod._HAS_AIOHTTP = True
                class DummySession:
                    async def __aenter__(self):
                        return self
                    async def __aexit__(self, exc_type, exc, tb):
                        pass
                di_mod.aiohttp = types.SimpleNamespace(ClientSession=DummySession)
                di_mod.download_triples(urls, urls, urls, root, ner_anonymizer=ner)
            t = Path(root) / 'text/0.txt'
            self.assertIn('[ORG]', t.read_text())
            cap = Path(root) / 'images/0.caption.txt'
            self.assertIn('[PERSON]', cap.read_text())
            trans = Path(root) / 'audio/0.transcript.txt'
            self.assertIn('[ORG]', trans.read_text())
            summ = ner.summary()
            self.assertGreaterEqual(summ.get('PERSON', 0), 2)
            self.assertGreaterEqual(summ.get('ORG', 0), 2)


if __name__ == '__main__':
    unittest.main()
