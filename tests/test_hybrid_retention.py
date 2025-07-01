import unittest
import torch

from asi.hybrid_retention import HybridRetention


class TestHybridRetention(unittest.TestCase):
    def test_forward_shape(self):
        module = HybridRetention(dim=8, num_heads=2)
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        out = module(q, k, v)
        self.assertEqual(out.shape, q.shape)


if __name__ == "__main__":
    unittest.main()
