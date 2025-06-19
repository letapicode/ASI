import unittest
import torch
from src.moe_router import HashRouter

class TestHashRouter(unittest.TestCase):
    def test_load_balance(self):
        router = HashRouter(num_experts=8)
        x = torch.randn(2, 512, 64)
        assignments = router(x)
        std = router.load_balance_std(assignments)
        self.assertLess(std, 0.03)
        util = router.expert_utilization(assignments)
        self.assertEqual(util.sum().item(), x.numel() // x.shape[-1] * router.k)

if __name__ == '__main__':
    unittest.main()
