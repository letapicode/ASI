import unittest
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']

try:
    import torch  # pragma: no cover - optional heavy dep
except Exception:  # pragma: no cover - torch unavailable
    torch = types.ModuleType('torch')
    torch.nn = types.SimpleNamespace(Module=object)
    torch.randn = lambda *a, **kw: None
    torch.zeros_like = lambda x: x
    torch.stack = lambda xs: xs
    torch.empty = lambda *a, **kw: None
    torch.Tensor = object
    torch.all = lambda *a, **kw: True
    import contextlib
    torch.no_grad = contextlib.nullcontext
    sys.modules['torch'] = torch


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    setattr(pkg, name.split('.')[-1], mod)
    return mod


goth = load('asi.graph_of_thought', 'src/graph_of_thought.py')
pr = load('asi.graph_pruning_manager', 'src/graph_pruning_manager.py')

GraphOfThought = goth.GraphOfThought
GraphPruningManager = pr.GraphPruningManager


class DummySummarizer:
    def summarize(self, text):
        return 'sum'

    def expand(self, text):
        return types.SimpleNamespace(unsqueeze=lambda dim: 0)


class DummyMemory:
    def __init__(self):
        self.summarizer = DummySummarizer()
        self.compressor = types.SimpleNamespace(encoder=lambda x: x)
        self.added = []
        self.translator = None

    def add_compressed(self, comp, meta):
        self.added.append((comp, meta))


class TestGraphPruningManager(unittest.TestCase):
    def test_prune_low_degree(self):
        mem = DummyMemory()
        g = GraphOfThought()
        a = g.add_step('a')
        b = g.add_step('b')
        c = g.add_step('c')
        g.connect(a, b)
        g.connect(b, c)
        pruner = GraphPruningManager(degree_threshold=1, memory=mem)
        pruner.attach(g)
        removed = pruner.prune_low_degree()
        self.assertIn(c, removed)
        self.assertNotIn(c, g.nodes)
        self.assertTrue(mem.added)

    def test_prune_old_nodes(self):
        mem = DummyMemory()
        g = GraphOfThought()
        old = g.add_step('old', timestamp=0.0)
        new = g.add_step('new', timestamp=160.0)
        pruner = GraphPruningManager(age_threshold=50.0, memory=mem)
        pruner.attach(g)
        removed = pruner.prune_old_nodes(now=200.0)
        self.assertIn(old, removed)
        self.assertNotIn(old, g.nodes)
        self.assertIn(new, g.nodes)


if __name__ == '__main__':
    unittest.main()
