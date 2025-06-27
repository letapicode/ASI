import os
import random
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from quantum_hpo import QAEHyperparamSearch


class TestQuantumHPO(unittest.TestCase):
    def test_search_selects_best_param(self):
        probs = {0: 0.1, 1: 0.5, 2: 0.9}

        def eval_func(p):
            return random.random() < probs[p]

        random.seed(0)
        search = QAEHyperparamSearch(eval_func, probs.keys())
        best_param, est = search.search(num_samples=3, shots=50)
        self.assertEqual(best_param, 2)
        self.assertGreater(est, 0.7)

    def test_search_bayesian_method(self):
        probs = {0: 0.2, 1: 0.4, 2: 0.8}

        def eval_func(p):
            return random.random() < probs[p]

        random.seed(2)
        search = QAEHyperparamSearch(eval_func, probs.keys())
        best_param, est = search.search(num_samples=3, shots=40, method="bayesian")
        self.assertEqual(best_param, 2)
        self.assertGreater(est, 0.6)


if __name__ == "__main__":
    unittest.main()
