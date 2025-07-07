import importlib.machinery
import importlib.util
import types
import sys
import unittest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    raise unittest.SkipTest("torch not available")

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


cs = load('asi.context_summary_memory', 'src/context_summary_memory.py')
di = load('asi.data_ingest', 'src/data_ingest.py')
goth = load('asi.graph_of_thought', 'src/graph_of_thought.py')
cg = load('asi.cross_lingual_graph', 'src/cross_lingual_graph.py')

CrossLingualReasoningGraph = cg.CrossLingualReasoningGraph
ContextSummaryMemory = cs.ContextSummaryMemory
CrossLingualTranslator = di.CrossLingualTranslator


class DummySummarizer:
    def summarize(self, text):
        return 'sum'

    def expand(self, text):
        return 0


class TestCrossLingualReasoningGraph(unittest.TestCase):
    def test_summarize_old_steps(self):
        tr = CrossLingualTranslator(['es'])
        mem = ContextSummaryMemory(
            dim=2,
            compressed_dim=1,
            capacity=4,
            summarizer=DummySummarizer(),
            translator=tr,
        )
        g = CrossLingualReasoningGraph(translator=tr)
        ids = [g.add_step(f's{i}') for i in range(5)]
        new = g.summarize_old_steps(ids, mem, threshold=2)
        self.assertEqual(len(new), 3)
        meta = g.nodes[new[0]].metadata
        self.assertIn('translations', meta)
        self.assertEqual(meta['translations']['es'], '[es] sum')


if __name__ == '__main__':
    unittest.main()
