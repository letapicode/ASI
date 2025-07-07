import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch
from torch import nn

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
loader = importlib.machinery.SourceFileLoader('asi.meta_optimizer', 'src/meta_optimizer.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
meta_mod = importlib.util.module_from_spec(spec)
meta_mod.__package__ = 'asi'
sys.modules['asi.meta_optimizer'] = meta_mod
loader.exec_module(meta_mod)
MetaOptimizer = meta_mod.MetaOptimizer


class TestMetaOptimizer(unittest.TestCase):
    def test_adaptation_improves_loss(self):
        torch.manual_seed(0)
        model = nn.Linear(1, 1, bias=False)

        def make_task(a: float):
            x = torch.randn(8, 1)
            y = a * x
            return (x, y)

        def train_step(m: nn.Module, data):
            x, y = data
            pred = m(x)
            return ((pred - y) ** 2).mean()

        opt = MetaOptimizer(train_step, adapt_lr=0.05, meta_lr=0.05, adapt_steps=1)
        tasks = [make_task(1.0), make_task(2.0)]
        for _ in range(5):
            opt.meta_step(model, tasks)

        new_task = make_task(1.5)
        before = ((model(new_task[0]) - new_task[1]) ** 2).mean().item()
        adapted = opt.adapt(model, new_task)
        after = ((adapted(new_task[0]) - new_task[1]) ** 2).mean().item()
        self.assertLess(after, before)


if __name__ == '__main__':
    unittest.main()
