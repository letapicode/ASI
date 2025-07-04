import unittest
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

loader = importlib.machinery.SourceFileLoader('src.graph_of_thought', 'src/graph_of_thought.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
got = importlib.util.module_from_spec(spec)
got.__package__ = 'src'
sys.modules['src.graph_of_thought'] = got
loader.exec_module(got)
GraphOfThought = got.GraphOfThought


class TestReasoningSummary(unittest.TestCase):
    def test_summarize(self):
        g = GraphOfThought()
        a = g.add_step('start')
        b = g.add_step('middle')
        c = g.add_step('end')
        g.connect(a, b)
        g.connect(b, c)
        path, summary = g.search(a, lambda n: n.id == c, explain=True)
        self.assertEqual(path, [a, b, c])
        self.assertEqual(summary, 'start -> middle -> end')


if __name__ == '__main__':
    unittest.main()
