import unittest
import torch

from asi.moe_layer import MoELayer
from asi.moe_router import HashRouter


class TestMoELayer(unittest.TestCase):
    def test_forward_shape(self):
        layer = MoELayer(dim=8, hidden=16, num_experts=4)
        x = torch.randn(2, 5, 8)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_balance_penalty(self):
        layer = MoELayer(dim=8, hidden=16, num_experts=4, router="switch", balance_weight=0.1)
        x = torch.randn(2, 5, 8)
        out, penalty = layer(x)
        self.assertEqual(out.shape, x.shape)
        self.assertGreaterEqual(penalty.item(), 0.0)

    def test_custom_router_instance(self):
        router = HashRouter(num_experts=4)
        layer = MoELayer(dim=8, hidden=16, num_experts=4, router=router)
        x = torch.randn(1, 3, 8)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_elastic_router_string(self):
        layer = MoELayer(dim=8, hidden=16, num_experts=4, router="elastic")
        x = torch.randn(1, 2, 8)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
