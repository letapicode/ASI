import unittest
import random
import torch

from asi.neural_arch_search import DistributedArchSearch


class TestNeuralArchSearch(unittest.TestCase):
    def test_search_selects_best(self):
        space = {"layers": [1, 2], "hidden": [8, 16]}
        random.seed(0)
        torch.manual_seed(0)

        def score(cfg):
            return cfg["layers"] * cfg["hidden"]

        search = DistributedArchSearch(space, score, max_workers=1)
        best, val = search.search(num_samples=4)
        self.assertEqual(best["layers"], 2)
        self.assertEqual(best["hidden"], 16)
        self.assertEqual(val, 32)


if __name__ == "__main__":
    unittest.main()
