import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

loader = importlib.machinery.SourceFileLoader('asi.sim2real_adapter', 'src/sim2real_adapter.py')
spec = importlib.util.spec_from_loader('asi.sim2real_adapter', loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = 'asi'
sys.modules['asi.sim2real_adapter'] = mod
loader.exec_module(mod)
setattr(pkg, 'sim2real_adapter', mod)
Sim2RealAdapter = mod.Sim2RealAdapter
Sim2RealConfig = mod.Sim2RealConfig


class TestSim2RealAdapter(unittest.TestCase):
    def test_fit(self):
        cfg = Sim2RealConfig(state_dim=2, epochs=20, lr=0.1)
        adapter = Sim2RealAdapter(cfg)
        logs = []
        for _ in range(50):
            s = torch.randn(2)
            logs.append((s, s + 1))
        adapter.fit(logs)
        out = adapter(torch.zeros(2))
        self.assertTrue(torch.allclose(out, torch.ones(2), atol=0.2))


if __name__ == '__main__':
    unittest.main()
