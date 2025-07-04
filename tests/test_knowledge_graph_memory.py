import unittest
import torch
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

KnowledgeGraphMemory = _load('asi.knowledge_graph_memory', 'src/knowledge_graph_memory.py').KnowledgeGraphMemory
_load('asi.streaming_compression', 'src/streaming_compression.py')
HierarchicalMemory = _load('asi.hierarchical_memory', 'src/hierarchical_memory.py').HierarchicalMemory


class TestKnowledgeGraphMemory(unittest.TestCase):
    def test_add_query(self):
        kg = KnowledgeGraphMemory()
        kg.add_triples([("a", "b", "c")])
        res = kg.query_triples(subject="a")
        self.assertIn(("a", "b", "c"), res)

    def test_hierarchical_integration(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10, use_kg=True)
        vec = torch.randn(1, 2)
        mem.add(vec, metadata=[("x", "y", "z")])
        q_vec, meta, triples = mem.search_with_kg(vec[0], k=1)
        self.assertEqual(len(triples), 1)
        self.assertIn(("x", "y", "z"), triples)


if __name__ == "__main__":
    unittest.main()
