import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import importlib.machinery
import importlib.util
import sys
import types

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

dv_loader = importlib.machinery.SourceFileLoader('src.dataset_versioner', 'src/dataset_versioner.py')
dv_spec = importlib.util.spec_from_loader(dv_loader.name, dv_loader)
dv = importlib.util.module_from_spec(dv_spec)
dv.__package__ = 'src'
sys.modules['src.dataset_versioner'] = dv
dv_loader.exec_module(dv)
src_pkg.dataset_versioner = dv

loader = importlib.machinery.SourceFileLoader('src.data_ingest', 'src/data_ingest.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
di = importlib.util.module_from_spec(spec)
di.__package__ = 'src'
sys.modules['src.data_ingest'] = di
sys.modules['asi.data_ingest'] = di
loader.exec_module(di)

CrossLingualTranslator = di.CrossLingualTranslator
download_triples = di.download_triples


class TestCrossLingualTranslator(unittest.TestCase):
    def test_translate_all(self):
        tr = CrossLingualTranslator(["en", "fr"])
        res = tr.translate_all("hello")
        self.assertEqual(res["en"], "[en] hello")
        self.assertIn("fr", res)

    def test_download_triples_translation(self):
        async def fake_download(session, url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("hello")

        with tempfile.TemporaryDirectory() as root:
            tr = CrossLingualTranslator(["es"])
            urls = ["u"]
            with patch.object(di, "_download_file_async", fake_download):
                triples = download_triples(urls, urls, urls, root, translator=tr)

            self.assertEqual(len(triples), 2)
            orig, translated = triples[0], triples[1]
            t_trans = Path(translated[0])
            self.assertTrue(t_trans.name.endswith("_es.txt"))
            self.assertTrue(t_trans.exists())
            self.assertEqual(t_trans.read_text(), "[es] hello")


if __name__ == "__main__":
    unittest.main()
