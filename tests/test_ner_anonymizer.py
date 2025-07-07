import unittest
import tempfile
import types
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg

# lightweight torch stub for data_ingest import
torch_stub = types.SimpleNamespace(
    tensor=lambda *a, **kw: None,
    zeros=lambda *a, **kw: None,
    long=lambda: None,
)
sys.modules['torch'] = torch_stub

requests_stub = types.SimpleNamespace(get=lambda *a, **k: types.SimpleNamespace(content=b'', raise_for_status=lambda: None))
sys.modules['requests'] = requests_stub

# psutil stub for CarbonFootprintTracker
psutil_stub = types.SimpleNamespace(cpu_percent=lambda interval=None: 0.0)
sys.modules['psutil'] = psutil_stub
sys.modules['pynvml'] = types.ModuleType('pynvml')

# load ner_anonymizer
loader_na = importlib.machinery.SourceFileLoader('src.ner_anonymizer', 'src/ner_anonymizer.py')
spec_na = importlib.util.spec_from_loader(loader_na.name, loader_na)
na_mod = importlib.util.module_from_spec(spec_na)
na_mod.__package__ = 'src'
sys.modules['src.ner_anonymizer'] = na_mod
loader_na.exec_module(na_mod)
NERAnonymizer = na_mod.NERAnonymizer

# load data_ingest
loader_di = importlib.machinery.SourceFileLoader('src.data_ingest', 'src/data_ingest.py')
spec_di = importlib.util.spec_from_loader(loader_di.name, loader_di)
di_mod = importlib.util.module_from_spec(spec_di)
di_mod.__package__ = 'src'
sys.modules['src.data_ingest'] = di_mod
loader_di.exec_module(di_mod)

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
        async def fake_download(session, url, dest):
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
