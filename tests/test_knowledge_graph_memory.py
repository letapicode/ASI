import unittest
import types
import sys


class FakeTensor:
    def __init__(self, shape=(1,)) -> None:
        self.shape = shape

    def dim(self) -> int:
        return len(self.shape)

    def unsqueeze(self, *a: int) -> "FakeTensor":
        return self

    def view(self, *a: int) -> "FakeTensor":
        return self

    def detach(self) -> "FakeTensor":
        return self

    def clone(self) -> "FakeTensor":
        return FakeTensor(self.shape)

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self):
        import numpy as np

        return np.zeros(self.shape, dtype=np.float32)

    def __iter__(self):
        for _ in range(self.shape[0]):
            yield FakeTensor(self.shape[1:])

    def __getitem__(self, idx):
        return FakeTensor(self.shape[1:])


class FakeLinear:
    def __init__(self, out_f: int) -> None:
        self.out = out_f

    def __call__(self, x: FakeTensor) -> FakeTensor:
        return FakeTensor((x.shape[0], self.out))


torch_stub = types.SimpleNamespace(
    tensor=lambda *a, **k: FakeTensor(tuple(a)),
    randn=lambda *a, **k: FakeTensor(tuple(a)),
    nn=types.SimpleNamespace(Module=object, Linear=lambda in_f, out_f: FakeLinear(out_f)),
    Tensor=FakeTensor,
)
sys.modules['torch'] = torch_stub
torch = torch_stub
import importlib.machinery
import importlib.util

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']

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
        mem = HierarchicalMemory(
            dim=2,
            compressed_dim=1,
            capacity=10,
            use_kg=True,
            encryption_key=b'0' * 16,
        )
        mem.kg.add_triples([("x", "y", "z")])
        triples = mem.query_triples(subject="x")
        self.assertEqual(len(triples), 1)
        t = triples[0]
        self.assertIsInstance(t, TimedTriple)
        self.assertEqual((t.subject, t.predicate, t.object), ("x", "y", "z"))


if __name__ == "__main__":
    unittest.main()
