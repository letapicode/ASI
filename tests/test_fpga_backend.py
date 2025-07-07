import unittest
import importlib.machinery
import importlib.util
import types
import sys
from unittest.mock import patch
import torch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
loader = importlib.machinery.SourceFileLoader('asi.fpga_backend', 'src/fpga_backend.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
fb = importlib.util.module_from_spec(spec)
fb.__package__ = 'asi'
sys.modules['asi.fpga_backend'] = fb
loader.exec_module(fb)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


class TestFPGABackend(unittest.TestCase):
    def test_compile_flag(self):
        model = DummyModel()
        with patch.object(fb, "_HAS_FPGA", True):
            accel = fb.FPGAAccelerator(model, forward_fn=model.forward)
            accel.compile()
            self.assertTrue(accel.compiled)

    def test_run_matches_cpu(self):
        model = DummyModel()
        accel = fb.FPGAAccelerator(model, forward_fn=model.forward)
        inp = torch.tensor([1.0])
        out = accel.run(inp)
        self.assertTrue(torch.allclose(out, model(inp)))


if __name__ == "__main__":
    unittest.main()
