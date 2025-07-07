import json
import subprocess
import sys
import tempfile
import unittest
import os
import importlib.machinery
import importlib.util
try:
    import torch
except Exception:  # pragma: no cover - optional heavy dep
    torch = None
import types
import sys

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

HAS_TORCH = getattr(torch, "__version__", None) is not None
if HAS_TORCH:
    GraphOfThought = _load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought
    ReasoningHistoryLogger = _load('asi.reasoning_history', 'src/reasoning_history.py').ReasoningHistoryLogger
    ContextSummaryMemory = _load('asi.context_summary_memory', 'src/context_summary_memory.py').ContextSummaryMemory
    TransformerCircuitAnalyzer = _load('asi.transformer_circuit_analyzer', 'src/transformer_circuit_analyzer.py').TransformerCircuitAnalyzer
else:
    class GraphOfThought:
        def __init__(self, *a, **kw):
            pass

    class ReasoningHistoryLogger:
        def __init__(self, *a, **kw):
            pass

    class ContextSummaryMemory:
        def __init__(self, *a, **kw):
            self.summarizer = types.SimpleNamespace(summarize=lambda x: "", expand=lambda x: 0)

    class TransformerCircuitAnalyzer:
        def __init__(self, *a, **kw):
            pass
        def head_importance(self, *a, **kw):
            class Dummy(list):
                def tolist(self):
                    return []

            return Dummy()


@unittest.skipIf(not HAS_TORCH, "torch not available")
class TestGraphOfThought(unittest.TestCase):
    def test_add_and_search(self):
        g = GraphOfThought()
        a = g.add_step("start")
        b = g.add_step("analyze")
        c = g.add_step("apply refactor")
        g.connect(a, b)
        g.connect(b, c)
        path = g.plan_refactor(a, keyword="refactor")
        self.assertEqual(path, [a, b, c])

    def test_unreachable(self):
        g = GraphOfThought()
        a = g.add_step("start")
        b = g.add_step("middle")
        g.connect(a, b)
        path = g.plan_refactor(a, keyword="refactor")
        self.assertEqual(path, [])

    def test_self_reflect_and_logger(self):
        g = GraphOfThought()
        a = g.add_step("start")
        b = g.add_step("middle")
        c = g.add_step("end")
        g.connect(a, b)
        g.connect(b, c)
        summary = g.self_reflect()
        self.assertEqual(summary, "start -> middle -> end")

        logger = ReasoningHistoryLogger()
        logger.log(summary)
        hist = logger.get_history()
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0][1], summary)

    def test_plan_refactor_with_summary(self):
        g = GraphOfThought()
        a = g.add_step("start")
        b = g.add_step("analyze")
        c = g.add_step("apply refactor")
        g.connect(a, b)
        g.connect(b, c)

        class DummySummarizer:
            def summarize(self, text):
                return "sum"

            def expand(self, text):
                return 0

        mem = ContextSummaryMemory(
            dim=2, compressed_dim=1, capacity=2, summarizer=DummySummarizer()
        )
        path, summary = g.plan_refactor(
            a, summary_memory=mem, summary_threshold=1
        )
        self.assertEqual(path, [a, b, c])
        self.assertEqual(summary, "sum")

    @unittest.skipIf(not HAS_TORCH, "torch not available")
    def test_add_step_logs_heads(self):
        layer = torch.nn.TransformerEncoderLayer(d_model=8, nhead=2)
        model = torch.nn.TransformerEncoder(layer, num_layers=1)
        analyzer = TransformerCircuitAnalyzer(model, "layers.0.self_attn")
        g = GraphOfThought(analyzer=analyzer, layer="layers.0.self_attn")
        sample = torch.randn(4, 3, 8)
        nid = g.add_step("start", sample=sample)
        self.assertIn("head_importance", g.nodes[nid].metadata)


@unittest.skipIf(not HAS_TORCH, "torch not available")
class TestGraphOfThoughtCLI(unittest.TestCase):
    def test_cli_runs(self):
        data = {
            "nodes": [
                {"id": 0, "text": "start"},
                {"id": 1, "text": "apply refactor"},
            ],
            "edges": [[0, 1]],
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            json.dump(data, f)
            fname = f.name
        try:
            proc = subprocess.run(
                [sys.executable, "src/graph_of_thought.py", fname, "0"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertIn("0 -> 1", proc.stdout)
        finally:
            os.unlink(fname)


if __name__ == "__main__":  # pragma: no cover - test helper
    unittest.main()
