import unittest
import tempfile
from pathlib import Path
import types
import json
import numpy as np
import wave
from unittest.mock import patch
import importlib.machinery
import importlib.util
import importlib
import sys

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)

# stub heavy deps
sys.modules['torch'] = types.SimpleNamespace()
sys.modules['psutil'] = types.SimpleNamespace()

loader_pg = importlib.machinery.SourceFileLoader('src.privacy', 'src/privacy.py')
spec_pg = importlib.util.spec_from_loader(loader_pg.name, loader_pg)
pg_mod = importlib.util.module_from_spec(spec_pg)
pg_mod.__package__ = 'src'
sys.modules['src.privacy'] = pg_mod
sys.modules['asi.privacy'] = pg_mod
loader_pg.exec_module(pg_mod)
PrivacyGuard = pg_mod.PrivacyGuard

loader_di = importlib.machinery.SourceFileLoader('src.data_ingest', 'src/data_ingest.py')
spec_di = importlib.util.spec_from_loader(loader_di.name, loader_di)
di = importlib.util.module_from_spec(spec_di)
di.__package__ = 'src'
sys.modules['src.data_ingest'] = di
loader_di.exec_module(di)
Image = importlib.import_module('PIL.Image')


class TestPrivacyGuard(unittest.TestCase):
    def test_inject_budget(self):
        pg = PrivacyGuard(1.0, noise_scale=0.5)
        txt, img, aud = pg.inject("hello world", np.zeros((2, 2)), np.zeros(4), epsilon=0.2)
        self.assertLess(pg.remaining_budget(), 1.0)
        self.assertEqual(pg.remaining_budget(), 0.8)
        changed = txt != "hello world" or not np.allclose(img, 0) or not np.allclose(aud, 0)
        self.assertTrue(changed)

    def test_download_triples_privacy(self):
        async def fake_download(session, url, dest, watermark_id=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.suffix == ".txt":
                dest.write_text("hello world")
            elif dest.suffix == ".png":
                Image.new("RGB", (1, 1)).save(dest)
            else:
                with wave.open(str(dest), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(np.zeros(1, dtype=np.int16).tobytes())

        with tempfile.TemporaryDirectory() as root:
            pg = PrivacyGuard(0.5, noise_scale=0.5)
            urls = ["u"]
            with patch.object(di, "_download_file_async", fake_download):
                di._HAS_AIOHTTP = True
                class DummySession:
                    async def __aenter__(self):
                        return self
                    async def __aexit__(self, exc_type, exc, tb):
                        pass
                di.aiohttp = types.SimpleNamespace(ClientSession=DummySession)
                di.download_triples(urls, urls, urls, root, privacy_guard=pg)
            meta = Path(root) / "text/0.json"
            self.assertTrue(meta.exists())
            info = json.loads(meta.read_text())
            self.assertIn("epsilon", info)
            self.assertLess(pg.remaining_budget(), 0.5)


if __name__ == "__main__":
    unittest.main()
