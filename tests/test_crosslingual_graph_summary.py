import importlib.util
import importlib.machinery
import types
import sys
import unittest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    raise unittest.SkipTest("torch not available")

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']


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
cg = load('asi.cross_lingual_graph', 'src/cross_lingual_graph.py')
rh = load('asi.reasoning_history', 'src/reasoning_history.py')

ContextSummaryMemory = cs.ContextSummaryMemory
CrossLingualTranslator = di.CrossLingualTranslator
CrossLingualReasoningGraph = cg.CrossLingualReasoningGraph
ReasoningHistoryLogger = rh.ReasoningHistoryLogger


class DummySummarizer:
    def summarize(self, text):
        return 'sum'

    def expand(self, text):
        return torch.ones(2)


class TestCrossLingualGraphSummary(unittest.TestCase):
    def test_query_summary_translations(self):
        tr = CrossLingualTranslator(['es', 'fr'])
        mem = ContextSummaryMemory(
            dim=2,
            compressed_dim=1,
            capacity=4,
            summarizer=DummySummarizer(),
            translator=tr,
            encryption_key=b'0'*16,
        )
        logger = ReasoningHistoryLogger(translator=tr)
        g = CrossLingualReasoningGraph(translator=tr, logger=logger)
        ids = [g.add_step(f's{i}') for i in range(4)]
        new = g.summarize_old_steps(ids, mem, threshold=2)
        self.assertEqual(len(new), 3)
        vec = mem.summarizer.expand('sum')
        _v, meta = mem.search(vec, k=1, language='fr')
        self.assertTrue(meta and meta[0].startswith('[fr]'))
        summary = g.query_summary([new[0]], mem, lang='es')
        self.assertEqual(summary, '[es] sum')
        self.assertTrue(logger.entries)
        entry = logger.entries[0][1]
        self.assertIn('nodes', entry)
        self.assertIn('location', entry)


if __name__ == '__main__':  # pragma: no cover - test helper
    unittest.main()
