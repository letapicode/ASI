import importlib.machinery
import importlib.util
import types
import sys
import unittest

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


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
clm = load('asi.cross_lingual_memory', 'src/cross_lingual_memory.py')
ar = load('asi.analogical_retrieval', 'src/analogical_retrieval.py')

CrossLingualTranslator = di.CrossLingualTranslator
CrossLingualMemory = clm.CrossLingualMemory
analogy_search = ar.analogy_search


class TestCrossLingualAnalogy(unittest.TestCase):
    def test_retrieval_across_languages(self):
        tr = CrossLingualTranslator(['es'])
        mem = CrossLingualMemory(dim=4, compressed_dim=2, capacity=10, translator=tr)
        words = ['man', 'woman', 'king', 'queen']
        mem.add_texts(words, metadata=words)
        vecs, meta = analogy_search(
            mem,
            ('king', 'en'),
            ('man', 'en'),
            ('woman', 'es'),
            k=1
        )
        self.assertEqual(len(meta), 1)
        self.assertIn(meta[0], ['queen', '[es] queen'])


if __name__ == '__main__':
    unittest.main()
