import unittest
from unittest.mock import patch
import importlib.machinery
import importlib.util
import types
import sys
import torch

pkg = types.ModuleType('asi')
sys_modules_backup = {}
for name in ['asi']:
    sys_modules_backup[name] = sys.modules.get(name)
    sys.modules[name] = pkg

loader = importlib.machinery.SourceFileLoader('asi.analog_backend', 'src/analog_backend.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
ab = importlib.util.module_from_spec(spec)
ab.__package__ = 'asi'
sys.modules['asi.analog_backend'] = ab
loader.exec_module(ab)


class TestAnalogBackend(unittest.TestCase):
    def test_matmul_offload(self):
        dummy = types.SimpleNamespace(matmul=lambda a, b, noise=0.0: a @ b + 1)
        with patch.object(ab, '_HAS_ANALOG', True), patch.object(ab, 'analogsim', dummy):
            accel = ab.AnalogAccelerator()
            x = torch.eye(2)
            y = torch.eye(2)
            out = accel.matmul(x, y)
            self.assertTrue(torch.allclose(out, dummy.matmul(x, y)))

    def test_fallback_cpu(self):
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        accel = ab.AnalogAccelerator()
        out = accel.matmul(x, y)
        self.assertTrue(torch.allclose(out, x @ y))

    def test_context_manager(self):
        dummy = types.SimpleNamespace(matmul=lambda a, b, noise=0.0: a @ b + 1)
        with patch.object(ab, '_HAS_ANALOG', True), patch.object(ab, 'analogsim', dummy):
            accel = ab.AnalogAccelerator()
            x = torch.eye(2)
            y = torch.eye(2)
            with accel:
                out = torch.matmul(x, y)
            self.assertTrue(torch.allclose(out, dummy.matmul(x, y)))


if __name__ == '__main__':
    unittest.main()
