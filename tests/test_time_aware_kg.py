import unittest
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

loader = importlib.machinery.SourceFileLoader('asi.knowledge_graph_memory', 'src/knowledge_graph_memory.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
kg_mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = kg_mod
loader.exec_module(kg_mod)
KnowledgeGraphMemory = kg_mod.KnowledgeGraphMemory
TimedTriple = kg_mod.TimedTriple


class TestTimeAwareKG(unittest.TestCase):
    def test_time_range_query(self):
        kg = KnowledgeGraphMemory()
        kg.add_triples([
            TimedTriple('a', 'r', 'b', 1.0),
            TimedTriple('a', 'r', 'c', 5.0),
            ('x', 'y', 'z'),
        ])

        all_a = kg.query_triples(subject='a')
        self.assertEqual(len(all_a), 2)

        mid = kg.query_triples(subject='a', start_time=2.0, end_time=6.0)
        self.assertEqual(len(mid), 1)
        self.assertEqual(mid[0].object, 'c')

        filtered = kg.query_triples(start_time=0)
        self.assertTrue(all(t.timestamp is not None for t in filtered))


if __name__ == '__main__':
    unittest.main()
