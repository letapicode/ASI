import unittest
import importlib.machinery
import importlib.util
import sys

loader = importlib.machinery.SourceFileLoader('graph_of_thought', 'src/graph_of_thought.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod
loader.exec_module(mod)
GraphOfThought = mod.GraphOfThought
ReasoningDebugger = mod.ReasoningDebugger


class TestReasoningDebugger(unittest.TestCase):
    def test_single_and_multi_agent(self):
        g1 = GraphOfThought()
        a1 = g1.add_step('start')
        b1 = g1.add_step('not start')
        g1.connect(a1, b1)
        g1.connect(b1, a1)

        g2 = GraphOfThought()
        c1 = g2.add_step('start')
        g2.connect(c1, c1)

        dbg = ReasoningDebugger({'one': g1, 'two': g2})
        loops = dbg.find_loops()
        self.assertIn('one', loops)
        self.assertIn('two', loops)
        self.assertTrue(any(loops.values()))

        contrad = dbg.find_contradictions()
        self.assertTrue(any(a != c for a, _, c, _ in contrad))
        self.assertIsInstance(dbg.report(), str)


if __name__ == '__main__':
    unittest.main()
