import unittest
import sys
try:
    import torch
except Exception:  # pragma: no cover - optional heavy dep
    torch = None
if torch is None:
    torch = types.ModuleType('torch')
    torch.nn = types.SimpleNamespace(
        Module=object,
        TransformerEncoderLayer=lambda *a, **kw: types.SimpleNamespace(),
        TransformerEncoder=lambda layer, num_layers=1: types.SimpleNamespace(
            layers=[layer]
        ),
        MultiheadAttention=lambda *a, **kw: types.SimpleNamespace(
            num_heads=1,
            head_dim=1,
            embed_dim=1,
            in_proj_weight=types.SimpleNamespace(grad=None),
            forward=lambda *a, **kw: (a[0], None),
        ),
    )
    torch.randn = lambda *a, **kw: None
    torch.zeros_like = lambda x: x
    torch.stack = lambda xs: xs
    torch.empty = lambda *a, **kw: None
    torch.Tensor = object
    torch.all = lambda x: True
    sys.modules['torch'] = torch
else:
    sys.modules['torch'] = torch
HAS_TORCH = getattr(torch, "__version__", None) is not None
import types
import importlib.machinery
import importlib.util

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
pkg.__spec__ = importlib.machinery.ModuleSpec('asi', None, is_package=True)
sys.modules['asi'] = pkg

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

TransformerCircuitAnalyzer = _load('asi.transformer_circuit_analyzer', 'src/transformer_circuit_analyzer.py').TransformerCircuitAnalyzer


class TestTransformerCircuitAnalyzer(unittest.TestCase):
    def setUp(self):
        layer = torch.nn.TransformerEncoderLayer(d_model=8, nhead=2)
        self.model = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.sample = torch.randn(4, 3, 8)

    @unittest.skipIf(not HAS_TORCH, "torch not available")
    def test_head_importance_methods(self):
        analyzer = TransformerCircuitAnalyzer(self.model, "layers.0.self_attn")
        g_imp = analyzer.head_importance(self.sample, method="gradient")
        a_imp = analyzer.head_importance(self.sample, method="ablation")
        self.assertEqual(g_imp.numel(), self.model.layers[0].self_attn.num_heads)
        self.assertEqual(a_imp.numel(), self.model.layers[0].self_attn.num_heads)
        self.assertTrue(torch.all(g_imp >= 0))
        self.assertTrue(torch.all(a_imp >= 0))


if __name__ == "__main__":
    unittest.main()
