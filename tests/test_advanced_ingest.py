import unittest
import tempfile
from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types
import json
from unittest import mock

sys.modules['torch'] = types.SimpleNamespace(
    tensor=lambda *a, **kw: None,
    zeros=lambda *a, **kw: None,
    long=lambda: None,
)
sys.modules['requests'] = types.ModuleType('requests')

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg

# Load modules dynamically
loader_di = importlib.machinery.SourceFileLoader('src.data_ingest', 'src/data_ingest.py')
spec_di = importlib.util.spec_from_loader(loader_di.name, loader_di)
di = importlib.util.module_from_spec(spec_di)
di.__package__ = 'src'
sys.modules['src.data_ingest'] = di
sys.modules['asi.data_ingest'] = di
loader_di.exec_module(di)

loader_ai = importlib.machinery.SourceFileLoader('src.advanced_ingest', 'src/advanced_ingest.py')
spec_ai = importlib.util.spec_from_loader(loader_ai.name, loader_ai)
ai = importlib.util.module_from_spec(spec_ai)
ai.__package__ = 'src'
sys.modules['src.advanced_ingest'] = ai
sys.modules['asi.advanced_ingest'] = ai
loader_ai.exec_module(ai)
LLMIngestParser = ai.LLMIngestParser

class TestAdvancedIngest(unittest.TestCase):
    def test_parser_fallback(self):
        parser = LLMIngestParser()
        triples = parser.parse('Alice loves Bob')
        self.assertEqual(triples, [('Alice', 'loves', 'Bob')])

    def test_parser_multi_sentence(self):
        parser = LLMIngestParser()
        txt = 'Alice loves Bob. Carol admires Dave.'
        triples = parser.parse(txt)
        self.assertIn(('Alice', 'loves', 'Bob'), triples)
        self.assertIn(('Carol', 'admires', 'Dave'), triples)

    def test_env_model_loading(self):
        calls = []
        ai._HAS_SPACY = True
        ai.spacy = types.SimpleNamespace(load=lambda name: calls.append(name) or None)
        with mock.patch.dict('os.environ', {"LLM_PARSER_MODEL": "custom"}):
            LLMIngestParser()
        self.assertIn('custom', calls)
        ai._HAS_SPACY = False

    def test_parse_file_cache(self):
        parser = LLMIngestParser()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 't.txt'
            p.write_text('Alice loves Bob')
            out = parser.parse_file_to_json(p)
            mtime = out.stat().st_mtime
            out2 = parser.parse_file_to_json(p)
            self.assertEqual(out, out2)
            self.assertEqual(out2.stat().st_mtime, mtime)

    def test_download_triples_llm(self):
        async def fake_download(session, url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text('Alice loves Bob')

        di._HAS_AIOHTTP = True
        class DummySession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
        di.aiohttp = types.SimpleNamespace(ClientSession=DummySession)

        with tempfile.TemporaryDirectory() as root:
            urls = ['u']
            with mock.patch.object(di, '_download_file_async', fake_download):
                triples = di.download_triples(urls, urls, urls, root, use_llm_parser=True)
            triple_file = Path(root) / 'text' / '0.triples.json'
            self.assertTrue(triple_file.exists())
            data = json.loads(triple_file.read_text())
            self.assertEqual(data[0], ['Alice', 'loves', 'Bob'])
            self.assertEqual(len(triples), 1)

if __name__ == '__main__':
    unittest.main()
