import importlib.machinery
import importlib.util
import sys
import types
import unittest
import torch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

loader = importlib.machinery.SourceFileLoader('asi.moe_router', 'src/moe_router.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
moe_router = importlib.util.module_from_spec(spec)
moe_router.__package__ = 'asi'
sys.modules['asi.moe_router'] = moe_router
loader.exec_module(moe_router)

RLMoERouter = moe_router.RLMoERouter


class TestRLMoERouter(unittest.TestCase):
    def test_balance_improves(self):
        router = RLMoERouter(dim=8, num_experts=4, lr=0.5)
        x = torch.randn(4, 4, 8)
        assign, _ = router(x)
        before = router.load_balance_std(assign)
        for _ in range(100):
            router(x)
        assign, _ = router(x)
        after = router.load_balance_std(assign)
        self.assertLess(after, before)

    def test_convergence(self):
        router = RLMoERouter(dim=8, num_experts=4, lr=0.5)
        x = torch.randn(4, 4, 8)
        for _ in range(200):
            router(x)
        assign, _ = router(x)
        std = router.load_balance_std(assign)
        self.assertLess(std, 0.4)


if __name__ == '__main__':
    unittest.main()
