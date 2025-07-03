import unittest
from asi.adversarial_robustness import AdversarialRobustnessSuite

class TestAdversarialRobustnessSuite(unittest.TestCase):
    def test_generate(self):
        def model(p: str) -> float:
            return float(len(p))
        suite = AdversarialRobustnessSuite(model)
        adv = suite.generate("hi", ["hi", "h", "hello"])
        self.assertEqual(adv, "h")

if __name__ == '__main__':
    unittest.main()
