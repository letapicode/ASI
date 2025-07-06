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
    sys.modules[name] = mod
    loader.exec_module(mod)
    setattr(pkg, name.split('.')[-1], mod)
    return mod


kg = load('asi.knowledge_graph_memory', 'src/knowledge_graph_memory.py')
di = load('asi.data_ingest', 'src/data_ingest.py')
kgm = load('asi.cross_lingual_kg_memory', 'src/cross_lingual_kg_memory.py')

CrossLingualKGMemory = kgm.CrossLingualKGMemory
CrossLingualTranslator = di.CrossLingualTranslator
TimedTriple = kgm.TimedTriple


class TestCrossLingualKGMemory(unittest.TestCase):
    def test_multilingual_add_query(self):
        tr = CrossLingualTranslator(['es'])
        kg = CrossLingualKGMemory(translator=tr)
        kg.add_triples_multilingual([('a', 'likes', 'b')])
        es = kg.query_triples(subject='[es] a')
        self.assertEqual(len(es), 1)
        res = kg.query_translated(subject='a', lang='es')
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].subject, '[es] a')

    def test_cross_query(self):
        tr = CrossLingualTranslator(['es'])
        kg = CrossLingualKGMemory(translator=tr)
        kg.add_triples_multilingual([('x', 'y', 'z')])
        res = kg.query_translated(subject='[es] x', lang='en')
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].subject, 'x')


if __name__ == '__main__':
    unittest.main()
