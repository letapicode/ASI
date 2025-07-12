import unittest
import torch

from asi.retention import RetNetRetention

class TestRetNetRetention(unittest.TestCase):
    def test_retention_shapes(self):
        module = RetNetRetention()
        q = torch.randn(2, 5, 4)
        k = torch.randn(2, 5, 4)
        v = torch.randn(2, 5, 4)
        out = module(q, k, v)
        self.assertEqual(out.shape, q.shape)

    def test_multihead_shapes(self):
        module = RetNetRetention(num_heads=2, decay=[0.9, 0.8])
        q = torch.randn(3, 4, 8)
        k = torch.randn(3, 4, 8)
        v = torch.randn(3, 4, 8)
        out = module(q, k, v)
        self.assertEqual(out.shape, q.shape)

if __name__ == '__main__':
    unittest.main()
