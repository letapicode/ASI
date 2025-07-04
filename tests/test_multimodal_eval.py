import unittest
import torch
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

asi_pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', asi_pkg)

loader = importlib.machinery.SourceFileLoader('cmf', 'src/cross_modal_fusion.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
cmf = importlib.util.module_from_spec(spec)
loader.exec_module(cmf)
sys.modules["asi.cross_modal_fusion"] = cmf
CrossModalFusionConfig = cmf.CrossModalFusionConfig
CrossModalFusion = cmf.CrossModalFusion
MultiModalDataset = cmf.MultiModalDataset

loader2 = importlib.machinery.SourceFileLoader('eval', 'src/eval_harness.py')
spec2 = importlib.util.spec_from_loader(loader2.name, loader2)
eval_mod = importlib.util.module_from_spec(spec2)
sys.modules[loader2.name] = eval_mod
loader2.exec_module(eval_mod)
MultiModalEval = eval_mod.MultiModalEval

class DummyMem:
    def __init__(self, dim, **kwargs):
        self.dim = dim
        self.vecs = []
        self.meta = []

    def add_multimodal(self, t, i, a, metas):
        v = (t + i + a) / 3.0
        self.vecs.extend(v)
        self.meta.extend(metas)

    def search(self, q, k=1):
        sims = [float((v @ q).sum()) for v in self.vecs]
        idx = int(max(range(len(sims)), key=lambda j: sims[j]))
        return self.vecs[idx][None], [self.meta[idx]]

eval_mod.HierarchicalMemory = DummyMem
sys.modules["asi.hierarchical_memory"] = types.ModuleType("asi.hierarchical_memory")
sys.modules["asi.hierarchical_memory"].HierarchicalMemory = DummyMem

def tok(text: str, vocab: int):
    return [ord(c) % vocab for c in text]

class TestMultiModalEval(unittest.TestCase):
    def test_run(self):
        cfg = CrossModalFusionConfig(vocab_size=50, text_dim=8, img_channels=3, audio_channels=1, latent_dim=4)
        model = CrossModalFusion(cfg)
        data = [
            ("aa", torch.randn(3, 16, 16), torch.randn(1, 32)),
            ("bb", torch.randn(3, 16, 16), torch.randn(1, 32)),
        ]
        ds = MultiModalDataset(data, lambda t: tok(t, cfg.vocab_size))
        evaler = MultiModalEval(model, ds, k=1, batch_size=1)
        stats = evaler.run()
        self.assertEqual(set(stats.keys()), {"text", "image", "audio"})

if __name__ == '__main__':
    unittest.main()
