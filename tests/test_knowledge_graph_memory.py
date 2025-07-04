import unittest
import types
import sys

torch_stub = types.SimpleNamespace(
    tensor=lambda *a, **k: None,
    randn=lambda *a, **k: None,
)
sys.modules['torch'] = torch_stub
import importlib.machinery
import importlib.util

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

kg_mod = _load('asi.knowledge_graph_memory', 'src/knowledge_graph_memory.py')
KnowledgeGraphMemory = kg_mod.KnowledgeGraphMemory
TimedTriple = kg_mod.TimedTriple
_load('asi.streaming_compression', 'src/streaming_compression.py')
HierarchicalMemory = _load('asi.hierarchical_memory', 'src/hierarchical_memory.py').HierarchicalMemory


class TestKnowledgeGraphMemory(unittest.TestCase):
    def test_add_query(self):
        kg = KnowledgeGraphMemory()
        kg.add_triples([("a", "b", "c")])
        res = kg.query_triples(subject="a")
        self.assertIsInstance(res[0], TimedTriple)
        self.assertEqual((res[0].subject, res[0].predicate, res[0].object), ("a", "b", "c"))

    def test_hierarchical_integration(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10, use_kg=True)
        vec = torch.randn(1, 2)
        mem.add(vec, metadata=[("x", "y", "z")])
        q_vec, meta, triples = mem.search_with_kg(vec[0], k=1)
        self.assertEqual(len(triples), 1)
        t = triples[0]
        self.assertIsInstance(t, TimedTriple)
        self.assertEqual((t.subject, t.predicate, t.object), ("x", "y", "z"))


if __name__ == "__main__":
    unittest.main()
