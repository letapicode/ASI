import unittest
import torch
from asi.moe_router import HashRouter, SwitchRouter

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
        assignments = router(x)
        self.assertEqual(assignments.shape, (2, 512, 2))
        std = router.load_balance_std(assignments)
        self.assertLess(std, 0.5)  # gating may be imbalanced but should compute

if __name__ == '__main__':
    unittest.main()
