import unittest
import numpy as np

from asi.causal_graph_learner import CausalGraphLearner


class TestCausalGraphLearner(unittest.TestCase):
    def test_fit_edges(self):
        transitions = []
        for i in range(5):
            s = np.array([i, i], dtype=float)
            ns = s + np.array([1.0, 0.0])
            transitions.append((s, 0, ns))
        learner = CausalGraphLearner(threshold=0.5)
        learner.fit(transitions)
        edges = learner.edges()
        self.assertTrue(any(dst == 0 for _, dst, _ in edges))


if __name__ == "__main__":
    unittest.main()
