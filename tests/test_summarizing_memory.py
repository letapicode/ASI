import importlib.machinery
import importlib.util
import types
import sys
import unittest
import torch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

hm = types.ModuleType('asi.hierarchical_memory')

class _Store:
    def __init__(self):
        self._meta = []
        self._vectors = []

    def __len__(self):
        return len(self._vectors)

    def add(self, vecs, meta):
        if not isinstance(vecs, list):
            vecs = [vecs]
        self._vectors.extend(vecs)
        self._meta.extend(list(meta))

    def delete(self, tag=None):
        if tag in self._meta:
            idx = self._meta.index(tag)
            self._meta.pop(idx)
            self._vectors.pop(idx)


class HierarchicalMemory:
    def __init__(self, dim, compressed_dim, capacity, **kwargs):
        self.compressor = types.SimpleNamespace(
            buffer=types.SimpleNamespace(data=[]),
            encoder=types.SimpleNamespace(out_features=compressed_dim),
        )
        self.store = _Store()

    def add(self, x, metadata=None):
        if metadata is None:
            metadata = [None] * len(x)
        self.compressor.buffer.data.extend(list(x))
        self.store.add([v.numpy() if isinstance(v, torch.Tensor) else v for v in x], metadata)

    def search(self, query, k=5, **kwargs):
        vecs = torch.stack(self.compressor.buffer.data[:k]) if self.compressor.buffer.data else torch.zeros((0, query.size(-1)))
        meta = self.store._meta[:k]
        return vecs, meta

    def __len__(self):
        return len(self.store)

    def delete(self, tag=None):
        self.store.delete(tag)

hm.HierarchicalMemory = HierarchicalMemory
sys.modules['asi.hierarchical_memory'] = hm
setattr(pkg, 'hierarchical_memory', hm)

loader_sm = importlib.machinery.SourceFileLoader('sm', 'src/summarizing_memory.py')
spec_sm = importlib.util.spec_from_loader(loader_sm.name, loader_sm)
sm = importlib.util.module_from_spec(spec_sm)
sm.__package__ = 'asi'
loader_sm.exec_module(sm)
sys.modules['asi.summarizing_memory'] = sm
setattr(pkg, 'summarizing_memory', sm)

SummarizingMemory = sm.SummarizingMemory


class DummySummarizer:
    def __call__(self, x):
        return "sum"


class TestSummarizingMemory(unittest.TestCase):
    def test_summarize(self):
        mem = SummarizingMemory(dim=2, compressed_dim=1, capacity=3)
        data = torch.randn(3, 2)
        mem.add(data, metadata=["a", "b", "c"])
        mem.summarize(DummySummarizer())
        self.assertEqual(len(mem), 3)


if __name__ == "__main__":
    unittest.main()
