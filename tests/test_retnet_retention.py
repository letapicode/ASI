import unittest
import torch

from asi.retnet_retention import RetNetRetention

class TestRetNetRetention(unittest.TestCase):
    def test_retention_shapes(self):
        module = RetNetRetention()
        q = torch.randn(2, 5, 4)
        k = torch.randn(2, 5, 4)
        v = torch.randn(2, 5, 4)
        out = module(q, k, v)
        self.assertEqual(out.shape, q.shape)

if __name__ == '__main__':
    unittest.main()
