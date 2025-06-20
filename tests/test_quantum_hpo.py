import random
import unittest

from src.quantum_hpo import QAEHyperparamSearch


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


if __name__ == "__main__":
    unittest.main()
