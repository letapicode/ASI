import unittest
import importlib.machinery
import importlib.util
import types
import sys
try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    class _Tensor(list):
        @property
        def ndim(self):
            return 2 if self and isinstance(self[0], list) else 1
        def unsqueeze(self, dim):
            return _Tensor([self])
        def tolist(self):
            return list(self)
        def __iter__(self):
            for x in list.__iter__(self):
                if isinstance(x, list):
                    yield _Tensor(x)
                else:
                    yield x

    def _full(val, *shape):
        if len(shape) == 1:
            return _Tensor([val for _ in range(shape[0])])
        rows = shape[0]
        cols = shape[1]
        return _Tensor([[val for _ in range(cols)] for _ in range(rows)])

    torch = types.SimpleNamespace(
        zeros=lambda *a, **k: _full(0.0, *a),
        ones=lambda *a, **k: _full(1.0, *a),
        tensor=lambda v, dtype=None: _Tensor(v),
        Tensor=_Tensor,
    )
    sys.modules['torch'] = torch

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

    def test_summarize(self):
        q = torch.zeros(1, 2)
        r = torch.ones(2, 2)
        scores = [0.9, 0.8]
        prov = ['a', 'b']
        summary = RetrievalExplainer.summarize(q, r, scores, prov)
        self.assertIn('a', summary)
        self.assertIn('0.9', summary)

    def test_summarize_multimodal(self):
        q = torch.zeros(1, 2)
        r = torch.ones(2, 2)
        scores = [0.5, 0.4]
        prov = [
            {'text': 'hello', 'image': 'img1.png', 'audio': 'a1.wav'},
            {'text': 'world', 'image': 'img2.png', 'audio': 'a2.wav'},
        ]
        summary = RetrievalExplainer.summarize_multimodal(q, r, scores, prov)
        self.assertIn('hello', summary)
        self.assertIn('img1.png', summary)
        self.assertIn('a1.wav', summary)

if __name__ == '__main__':
    unittest.main()
