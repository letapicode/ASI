import importlib.machinery
import importlib.util
import types
import sys
import unittest

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

GraphOfThought = _load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought
merge_graphs = _load('asi.reasoning_merger', 'src/reasoning_merger.py').merge_graphs


class TestReasoningMerger(unittest.TestCase):
    def test_merge_order(self):
        g1 = GraphOfThought()
        a = g1.add_step('start', metadata={'timestamp': 0.0})
        b = g1.add_step('analyse', metadata={'timestamp': 1.0})
        g1.connect(a, b)

        g2 = GraphOfThought()
        b2 = g2.add_step('analyse', metadata={'timestamp': 1.0})
        c = g2.add_step('finish', metadata={'timestamp': 2.0})
        g2.connect(b2, c)

        merged, inconsist = merge_graphs({'g1': g1, 'g2': g2})
        texts = [merged.nodes[i].text for i in sorted(merged.nodes)]
        self.assertEqual(texts, ['start', 'analyse', 'finish'])
        self.assertEqual(merged.edges.get(0), [1])
        self.assertEqual(merged.edges.get(1), [2])
        self.assertFalse(inconsist)

    def test_inconsistency(self):
        g1 = GraphOfThought()
        g1.add_step('x', metadata={'timestamp': 0.0})
        g2 = GraphOfThought()
        g2.add_step('y', metadata={'timestamp': 0.0})
        _, inconsist = merge_graphs({'a': g1, 'b': g2})
        self.assertTrue(inconsist)


if __name__ == '__main__':  # pragma: no cover - test helper
    unittest.main()
