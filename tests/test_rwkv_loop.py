import unittest
import torch

from asi.rwkv_loop import RWKVLoop


class TestRWKVLoop(unittest.TestCase):
    def test_forward_shape(self):
        block = RWKVLoop(dim=4)
        x = torch.randn(2, 5, 4)
        out = block(x)
        self.assertEqual(out.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
