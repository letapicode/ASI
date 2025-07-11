import unittest
import http.client
import json
import sys
import types
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

if HAS_TORCH:
    InterpretabilityDashboard = _load('asi.interpretability_dashboard', 'src/interpretability_dashboard.py').InterpretabilityDashboard
    TransformerCircuitAnalyzer = _load('asi.transformer_circuit_analyzer', 'src/transformer_circuit_analyzer.py').TransformerCircuitAnalyzer
    GraphOfThought = _load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought
    TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
else:
    InterpretabilityDashboard = type('Dash', (), {'__init__': lambda self,*a, **k: None, 'start': lambda self, **kw: None, 'stop': lambda self: None, 'port': 0})
    TransformerCircuitAnalyzer = type('Analyzer', (), {'__init__': lambda self,*a, **k: None, 'head_importance': lambda self,*a, **k: []})
    GraphOfThought = type('Graph', (), {'__init__': lambda self,*a, **k: None, 'add_step': lambda self,*a, **k: 0, 'nodes': {}})
    TelemetryLogger = type('Tel', (), {'__init__': lambda self,*a, **k: None, 'start': lambda self: None, 'stop': lambda self: None})


class Mem:
    def get_stats(self):
        return {"hit_rate": 1.0}


class StubServer:
    def __init__(self, mem, tel):
        self.memory = mem
        self.telemetry = tel


class TestInterpretabilityDashboard(unittest.TestCase):
    @unittest.skipIf(not HAS_TORCH, "torch not available")
    def test_endpoints(self):
        mem = Mem()
        logger = TelemetryLogger(interval=0.1)
        logger.start()
        server = StubServer(mem, logger)
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=8, nhead=2), num_layers=1
        )
        sample = torch.randn(4, 1, 8)
        analyzer = TransformerCircuitAnalyzer(model, "layers.0.self_attn")
        graph = GraphOfThought(analyzer=analyzer, layer="layers.0.self_attn")
        graph.add_step("start", sample=sample)
        dash = InterpretabilityDashboard(model, [server], sample, graph=graph)
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", "/stats")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        self.assertIn("hit_rate", data)
        self.assertIn("head_contributions", data)
        conn.request("GET", "/heatmaps")
        resp = conn.getresponse()
        heat = json.loads(resp.read())
        self.assertIn("images", heat)
        dash.stop()
        logger.stop()


if __name__ == "__main__":
    unittest.main()
