import unittest
import torch

from src.moe_layer import MoELayer


class TestMoELayer(unittest.TestCase):
    def test_forward_shape(self):
        layer = MoELayer(dim=8, hidden=16, num_experts=4)
        x = torch.randn(2, 5, 8)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
