import unittest
import torch

from asi.hybrid_retention import HybridRetention


class TestHybridRetention(unittest.TestCase):
    def test_forward_shape(self):
        module = HybridRetention(dim=4)
        q = torch.randn(2, 5, 4)
        k = torch.randn(2, 5, 4)
        v = torch.randn(2, 5, 4)
        out = module(q, k, v)
        self.assertEqual(out.shape, q.shape)


if __name__ == "__main__":
    unittest.main()
