import importlib.machinery
import importlib.util
import types
import sys
import unittest

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']

loader = importlib.machinery.SourceFileLoader('asi.forecast_strategies', 'src/forecast_strategies.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod
loader.exec_module(mod)
_TrendTransformer = mod._TrendTransformer

try:
    import torch
except Exception:
    torch = None


class TestForecastStrategies(unittest.TestCase):
    def test_trend_transformer_forward(self):
        if torch is None:
            self.skipTest('torch not available')
        model = _TrendTransformer(hidden_size=4, nhead=1, num_layers=1)
        x = torch.zeros(1, 2, 2)
        out = model(x)
        self.assertEqual(list(out.shape), [1, 2])


if __name__ == '__main__':
    unittest.main()
