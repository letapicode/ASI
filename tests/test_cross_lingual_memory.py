import unittest
import importlib.machinery
import importlib.util
import sys
import types

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']

def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

di = load('asi.data_ingest', 'src/data_ingest.py')
hm = load('asi.hierarchical_memory', 'src/hierarchical_memory.py')
clm = load('asi.cross_lingual_memory', 'src/cross_lingual_memory.py')

CrossLingualMemory = clm.CrossLingualMemory
CrossLingualTranslator = di.CrossLingualTranslator

class TestCrossLingualMemory(unittest.TestCase):
    def test_retrieval_across_languages(self):
        tr = CrossLingualTranslator(["es"])
        mem = CrossLingualMemory(
            dim=4,
            compressed_dim=2,
            capacity=10,
            translator=tr,
            encryption_key=b'0'*16
        )
        mem.add("hello")
        vecs, meta = mem.search("[es] hello", k=1)
        self.assertEqual(len(meta), 1)
        self.assertIn(meta[0], ["hello", "[es] hello"])

if __name__ == "__main__":
    unittest.main()
