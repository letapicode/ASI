import unittest
import tempfile
import os
from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types
from asi.enclave_runner import EnclaveRunner

# Load data_ingest module similar to other tests
torch = types.SimpleNamespace(tensor=lambda *a, **kw: None,
                              zeros=lambda *a, **kw: None,
                              long=lambda: None)
sys.modules['torch'] = torch

loader = importlib.machinery.SourceFileLoader('src.data_ingest', 'src/data_ingest.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
di = importlib.util.module_from_spec(spec)
di.__package__ = 'src'
sys.modules['src.data_ingest'] = di
sys.modules['asi.data_ingest'] = di
loader.exec_module(di)

class TestEnclaveIngest(unittest.TestCase):
    def test_download_triples_enclave(self):
        async def fake_download(session, url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(os.environ.get('IN_ENCLAVE', '0'))

        di._HAS_AIOHTTP = True
        class DummySession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
        di.aiohttp = types.SimpleNamespace(ClientSession=DummySession)

        with tempfile.TemporaryDirectory() as root:
            with unittest.mock.patch.object(di, '_download_file_async', fake_download):
                runner = EnclaveRunner()
                triples = di.download_triples(['u'], ['u'], ['u'], root, runner=runner)
            t, _, _ = triples[0]
            self.assertEqual(Path(t).read_text(), '1')

    def test_paraphrase_multilingual_enclave(self):
        class Insp:
            def inspect(self, meta):
                Path(meta).with_name('flag.txt').write_text(os.environ.get('IN_ENCLAVE','0'))
                return True
        class Lin:
            def record(self, *a, **k):
                Path(a[0][0]).with_name('lin.txt').write_text(os.environ.get('IN_ENCLAVE','0'))
        with tempfile.TemporaryDirectory() as root:
            src = Path(root)/'x.txt'
            src.write_text('hi')
            meta = src.with_suffix('.json')
            meta.write_text('{}')
            insp = Insp()
            lin = Lin()
            trans = di.CrossLingualTranslator(['es'])
            runner = EnclaveRunner()
            out = di.paraphrase_multilingual([src], trans, None, insp, lin, runner=runner)
            self.assertTrue(out)
            self.assertEqual((src.with_name('flag.txt')).read_text(), '1')
            self.assertEqual((src.with_name('lin.txt')).read_text(), '1')

if __name__ == '__main__':
    unittest.main()
