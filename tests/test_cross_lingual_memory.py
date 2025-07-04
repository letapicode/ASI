import unittest
import importlib.machinery
import importlib.util
import sys

# Load modules similar to other tests using dynamic import
loader = importlib.machinery.SourceFileLoader('clm', 'src/cross_lingual_memory.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
clm = importlib.util.module_from_spec(spec)
clm.__package__ = 'asi'
loader.exec_module(clm)
sys.modules['asi.cross_lingual_memory'] = clm

loader2 = importlib.machinery.SourceFileLoader('di', 'src/data_ingest.py')
spec2 = importlib.util.spec_from_loader(loader2.name, loader2)
di = importlib.util.module_from_spec(spec2)
di.__package__ = 'asi'
loader2.exec_module(di)
sys.modules['asi.data_ingest'] = di

CrossLingualMemory = clm.CrossLingualMemory
CrossLingualTranslator = di.CrossLingualTranslator


class TestCrossLingualMemory(unittest.TestCase):
    def test_retrieval_across_languages(self):
        tr = CrossLingualTranslator(["es"])
        mem = CrossLingualMemory(dim=4, compressed_dim=2, capacity=10, translator=tr)
        mem.add("hello")
        vecs, meta = mem.search("[es] hello", k=1)
        self.assertEqual(len(meta), 1)
        self.assertIn(meta[0], ["hello", "[es] hello"])


if __name__ == "__main__":
    unittest.main()
