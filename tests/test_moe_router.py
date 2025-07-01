import unittest
import torch
from asi.moe_router import (
    BaseRouter,
    HashRouter,
    SwitchRouter,
    balance_loss_probs,
    token_drop_rate,
)
from asi.elastic_moe_router import ElasticMoERouter


class TestBaseRouter(unittest.TestCase):
    def test_subclasses(self):
        self.assertTrue(issubclass(HashRouter, BaseRouter))
        self.assertTrue(issubclass(SwitchRouter, BaseRouter))
        self.assertTrue(issubclass(ElasticMoERouter, BaseRouter))

class TestHashRouter(unittest.TestCase):
    def test_load_balance(self):
        router = HashRouter(num_experts=8)
        x = torch.randn(2, 512, 64)
        assignments = router(x)
        std = router.load_balance_std(assignments)
        self.assertLess(std, 0.03)
        util = router.expert_utilization(assignments)
        self.assertEqual(util.sum().item(), x.numel() // x.shape[-1] * router.k)

    def test_switch_router(self):
        router = SwitchRouter(dim=64, num_experts=8, k=2)
        x = torch.randn(2, 512, 64)
        assignments, weights = router(x)
        self.assertEqual(assignments.shape, (2, 512, 2))
        self.assertEqual(weights.shape, (2, 512, 2))
        std = router.load_balance_std(assignments)
        self.assertLess(std, 0.5)  # gating may be imbalanced but should compute

    def test_temperature_routing(self):
        router = SwitchRouter(dim=32, num_experts=4, k=1, temperature=0.5)
        x = torch.randn(1, 10, 32)
        assign, _ = router(x)
        self.assertEqual(assign.shape, (1, 10, 1))

    def test_elastic_router_adjustment(self):
        router = ElasticMoERouter(dim=16, num_experts=8, utilization_fn=lambda: 0.95)
        x = torch.randn(1, 5, 16)
        assign, _ = router(x)
        self.assertLess(router.active_experts, router.num_experts)
        self.assertEqual(assign.shape, (1, 5, router.k))

    def test_elastic_low_util(self):
        router = ElasticMoERouter(dim=16, num_experts=4, utilization_fn=lambda: 0.1)
        x = torch.randn(1, 3, 16)
        assign, _ = router(x)
        self.assertEqual(router.active_experts, router.num_experts)
        self.assertEqual(assign.shape, (1, 3, router.k))

    def test_token_drop_and_balance_probs(self):
        router = HashRouter(num_experts=4)
        x = torch.randn(1, 16, 8)
        assign = router(x)
        drop = token_drop_rate(assign, num_experts=4, capacity=2)
        self.assertGreaterEqual(drop, 0.0)
        self.assertLessEqual(drop, 1.0)

        logits = torch.randn(2, 3, 5)
        probs = torch.softmax(logits, dim=-1)
        loss = balance_loss_probs(probs)
        self.assertGreaterEqual(loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
