import importlib.machinery
import importlib.util
import unittest
import numpy as np

loader = importlib.machinery.SourceFileLoader('qmm', 'src/quantum_multimodal_retrieval.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
qmm = importlib.util.module_from_spec(spec)
qmm.__package__ = 'asi'
loader.exec_module(qmm)
import sys
sys.modules['asi.quantum_multimodal_retrieval'] = qmm
quantum_crossmodal_search = qmm.quantum_crossmodal_search
torch = qmm.torch


class SimpleMemory:
    def __init__(self, dim):
        self.vecs = []
        self.meta = []
        self.dim = dim

    def __len__(self):
        return len(self.vecs)

    def add_multimodal(self, text, image, audio, meta):
        fused = (text + image + audio) / 3.0
        for v, m in zip(fused, meta):
            self.vecs.append(torch.from_numpy(np.asarray(v)))
            self.meta.append(m)

    def search(self, q, k=5, return_scores=False):
        if not self.vecs:
            empty = torch.empty(0, self.dim)
            return (empty, [], []) if return_scores else (empty, [])
        mat = torch.stack(self.vecs)
        scores = (mat @ q).tolist()
        idx = np.argsort(scores)[::-1][:k]
        vecs = torch.stack([self.vecs[i] for i in idx])
        metas = [self.meta[i] for i in idx]
        if return_scores:
            return vecs, metas, [scores[i] for i in idx]
        return vecs, metas


class TestQuantumMultimodal(unittest.TestCase):
    def test_consistent_results(self):
        rng = np.random.default_rng(0)
        dim = 4
        texts = rng.normal(size=(5, dim)).astype(np.float32)
        images = rng.normal(size=(5, dim)).astype(np.float32)
        audios = rng.normal(size=(5, dim)).astype(np.float32)
        mem = SimpleMemory(dim)
        mem.add_multimodal(texts, images, audios, list(range(5)))
        queries = (texts + images + audios) / 3.0
        for idx, q in enumerate(queries):
            q = torch.from_numpy(q + rng.normal(scale=0.01, size=dim).astype(np.float32))
            np.random.seed(42)
            _, m1 = quantum_crossmodal_search(q, mem, k=2)
            np.random.seed(42)
            _, m2 = quantum_crossmodal_search(q, mem, k=2)
            self.assertEqual(m1, m2)


if __name__ == '__main__':
    unittest.main()
