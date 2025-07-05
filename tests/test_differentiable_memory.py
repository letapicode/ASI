import unittest
import importlib.machinery
import importlib.util
import sys
import types
import torch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

load('asi.streaming_compression', 'src/streaming_compression.py')
load('asi.hierarchical_memory', 'src/hierarchical_memory.py')
dm = load('asi.differentiable_memory', 'src/differentiable_memory.py')
DifferentiableMemory = dm.DifferentiableMemory


class TestDifferentiableMemory(unittest.TestCase):
    def test_gradient_update(self):
        mem = DifferentiableMemory(dim=3, compressed_dim=2, capacity=10)
        vec = torch.randn(1, 3)
        mem.add(vec, metadata=["a"])
        opt = torch.optim.SGD(mem.parameters(), lr=0.1)
        target = torch.ones(3)
        query = vec.clone().requires_grad_(True)

        out, _ = mem.search(query, k=1)
        loss = torch.nn.functional.mse_loss(out.squeeze(0), target)
        loss.backward()
        opt.step()

        out2, _ = mem.search(query, k=1)
        loss2 = torch.nn.functional.mse_loss(out2.squeeze(0), target)
        self.assertLess(loss2.item(), loss.item())


if __name__ == "__main__":
    unittest.main()
