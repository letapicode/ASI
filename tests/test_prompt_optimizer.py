import unittest
from asi.prompt_optimizer import PromptOptimizer

class TestPromptOptimizer(unittest.TestCase):
    def test_optimize(self):
        def scorer(p: str) -> float:
            return -len(p)
        opt = PromptOptimizer(scorer, "hello")
        res = opt.optimize(steps=5)
        self.assertIsInstance(res, str)
        self.assertTrue(len(opt.history) >= 1)

if __name__ == '__main__':
    unittest.main()
