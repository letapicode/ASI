import unittest
import torch

from asi.retention import HybridRetention


class TestHybridRetention(unittest.TestCase):
    def test_forward_shape(self):
        module = HybridRetention(num_heads=1, dim=4)
        q = torch.randn(2, 5, 4)
        k = torch.randn(2, 5, 4)
        v = torch.randn(2, 5, 4)
        out = module(q, k, v)
        self.assertEqual(out.shape, q.shape)

    def test_multihead_shapes(self):
        module = HybridRetention(num_heads=2, dim=8, decay=[0.9, 0.8])
        q = torch.randn(3, 4, 8)
        k = torch.randn(3, 4, 8)
        v = torch.randn(3, 4, 8)
        out = module(q, k, v)
        self.assertEqual(out.shape, q.shape)


if __name__ == "__main__":
    unittest.main()
