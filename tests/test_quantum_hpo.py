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

    def test_search_parallel_execution(self):
        probs = {0: 0.3, 1: 0.6, 2: 0.9}

        def eval_func(p):
            return random.random() < probs[p]

        random.seed(1)
        search = QAEHyperparamSearch(eval_func, probs.keys())
        best_param, est = search.search(num_samples=3, shots=30, max_workers=2)
        self.assertEqual(best_param, 2)
        self.assertGreaterEqual(est, 0.7)

    def test_search_bayesian_method(self):
        probs = {0: 0.2, 1: 0.4, 2: 0.8}

        def eval_func(p):
            return random.random() < probs[p]

        random.seed(2)
        search = QAEHyperparamSearch(eval_func, probs.keys())
        best_param, est = search.search(num_samples=3, shots=40, method="bayesian")
        self.assertEqual(best_param, 2)
        self.assertGreater(est, 0.6)

    def test_search_early_stop(self):
        probs = {0: False, 1: True, 2: False}
        calls = {"n": 0}

        def eval_func(p):
            calls["n"] += 1
            return probs[p]

        random.seed(0)
        search = QAEHyperparamSearch(eval_func, probs.keys())
        best_param, est = search.search(num_samples=3, shots=1, early_stop=0.8)
        self.assertEqual(best_param, 1)
        self.assertEqual(calls["n"], 1)
        self.assertGreaterEqual(est, 0.8)

    def test_search_with_arch_space(self):
        params = [0.5, 1.0]
        arch_probs = {"a": 0.3, "b": 0.7}

        def eval_func(arch, p):
            return random.random() < arch_probs[arch] * p

        random.seed(3)
        search = QAEHyperparamSearch(eval_func, params, arch_probs.keys())
        (arch, param), prob = search.search(num_samples=4, shots=40)
        self.assertEqual(arch, "b")
        self.assertEqual(param, 1.0)
        self.assertGreater(prob, 0.5)


if __name__ == "__main__":
    unittest.main()
