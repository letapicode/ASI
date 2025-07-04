import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.retrieval_explainer', 'src/retrieval_explainer.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = 'src'
sys.modules['src.retrieval_explainer'] = mod
loader.exec_module(mod)
RetrievalExplainer = mod.RetrievalExplainer

class TestRetrievalExplainer(unittest.TestCase):
    def test_format(self):
        q = torch.zeros(1,2)
        r = torch.ones(2,2)
        scores = [0.9, 0.8]
        prov = ['a','b']
        items = RetrievalExplainer.format(q, r, scores, prov)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]['provenance'], 'a')

if __name__ == '__main__':
    unittest.main()
