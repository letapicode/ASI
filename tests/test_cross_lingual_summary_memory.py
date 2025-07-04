import unittest
import importlib.machinery
import importlib.util
import types
import sys
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


di = load('asi.data_ingest', 'src/data_ingest.py')
cs = load('asi.context_summary_memory', 'src/context_summary_memory.py')

CrossLingualTranslator = di.CrossLingualTranslator
ContextSummaryMemory = cs.ContextSummaryMemory


class DummySummarizer:
    def summarize(self, x):
        return 'hello'

    def expand(self, text):
        return torch.ones(2)


class TestCrossLingualSummaryMemory(unittest.TestCase):
    def test_add_and_search_translated(self):
        tr = CrossLingualTranslator(['es', 'fr'])
        mem = ContextSummaryMemory(
            dim=2,
            compressed_dim=1,
            capacity=4,
            summarizer=DummySummarizer(),
            context_size=1,
            translator=tr,
        )
        data = torch.randn(3, 2)
        mem.add(data, metadata=['a', 'b', 'c'])
        mem.summarize_context()
        vecs, meta = mem.search(data[0], k=2, language='es')
        self.assertTrue(any(isinstance(m, str) and m.startswith('[es]') for m in meta))
        self.assertEqual(vecs.shape[0], len(meta))


if __name__ == '__main__':
    unittest.main()
