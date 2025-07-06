import importlib.machinery
import importlib.util
import sys
import types
import unittest

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

MultiAgentCoordinator = _load('asi.multi_agent_coordinator', 'src/multi_agent_coordinator.py').MultiAgentCoordinator
GraphOfThought = _load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought
cons_mod = _load('asi.consensus_reasoner', 'src/consensus_reasoner.py')
compute_consensus = cons_mod.compute_consensus
report_disagreements = cons_mod.report_disagreements


class DummyAgent:
    def __init__(self, graph):
        self.graph = graph

    def select_action(self, state):
        return 'act'

    def update(self, state, action, reward, next_state):
        pass

    def train(self, entries):
        pass


class TestConsensusReasoner(unittest.TestCase):
    def test_compute_no_conflict(self):
        g1 = GraphOfThought()
        a = g1.add_step('start', metadata={'timestamp': 0.0})
        b = g1.add_step('end', metadata={'timestamp': 1.0})
        g1.connect(a, b)

        g2 = GraphOfThought()
        a2 = g2.add_step('start', metadata={'timestamp': 0.0})
        b2 = g2.add_step('end', metadata={'timestamp': 1.0})
        g2.connect(a2, b2)

        coord = MultiAgentCoordinator({'a': DummyAgent(g1), 'b': DummyAgent(g2)})
        merged, inconsist = compute_consensus(coord)
        self.assertEqual(len(inconsist), 0)
        self.assertEqual(len(merged.nodes), 2)

    def test_compute_conflict(self):
        g1 = GraphOfThought()
        g1.add_step('x', metadata={'timestamp': 0.0})
        g2 = GraphOfThought()
        g2.add_step('y', metadata={'timestamp': 0.0})
        coord = MultiAgentCoordinator({'a': DummyAgent(g1), 'b': DummyAgent(g2)})
        _merged, inconsist = compute_consensus(coord)
        self.assertTrue(inconsist)
        rep = report_disagreements(inconsist)
        self.assertIn('timestamp', rep)


if __name__ == '__main__':
    unittest.main()
