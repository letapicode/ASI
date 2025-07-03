import unittest
from asi.graph_of_thought import GraphOfThought, ReasoningDebugger

class TestReasoningDebugger(unittest.TestCase):
    def test_debug(self):
        g = GraphOfThought()
        a = g.add_step("start")
        b = g.add_step("not start")
        g.connect(a, b)
        dbg = ReasoningDebugger(g)
        self.assertTrue(dbg.find_contradictions())
        self.assertTrue(dbg.find_loops())

if __name__ == '__main__':
    unittest.main()
