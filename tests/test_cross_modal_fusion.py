import unittest
import importlib.machinery
import importlib.util
import sys
import torch

loader = importlib.machinery.SourceFileLoader('cmf', 'src/cross_modal_fusion.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
cmf = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = cmf
sys.modules['asi.cross_modal_fusion'] = cmf
loader.exec_module(cmf)
CrossModalFusionConfig = cmf.CrossModalFusionConfig
CrossModalFusion = cmf.CrossModalFusion
MultiModalDataset = cmf.MultiModalDataset
encode_all = cmf.encode_all
train_fusion_model = cmf.train_fusion_model

from asi.hierarchical_memory import HierarchicalMemory

def simple_tokenizer(text: str):
    return [ord(c) % 50 for c in text]

class TestCrossModalFusion(unittest.TestCase):
    def setUp(self):
        cfg = CrossModalFusionConfig(
            vocab_size=50,
            text_dim=8,
            img_channels=3,
            audio_channels=1,
            latent_dim=4,
        )
        self.model = CrossModalFusion(cfg)

    def test_forward_all_modalities(self):
        tokens = torch.randint(0, 50, (2, 5))
        images = torch.randn(2, 3, 16, 16)
        audio = torch.randn(2, 1, 32)
        t, i, a = self.model(tokens, images, audio)
        self.assertEqual(t.shape, (2, 4))
        self.assertEqual(i.shape, (2, 4))
        self.assertEqual(a.shape, (2, 4))

    def test_forward_missing_modalities(self):
        tokens = torch.randint(0, 50, (1, 5))
        t, i, a = self.model(text=tokens)
        self.assertIsNotNone(t)
        self.assertIsNone(i)
        self.assertIsNone(a)

    def test_encode_and_store(self):
        triples = [
            ("hello", torch.randn(3, 16, 16), torch.randn(1, 32)),
            ("world", torch.randn(3, 16, 16), torch.randn(1, 32)),
        ]
        ds = MultiModalDataset(triples, simple_tokenizer)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        vecs = encode_all(self.model, ds, batch_size=1, memory=mem)
        self.assertEqual(vecs[0].shape[0], len(ds))
        q = vecs[0][0]
        out, meta = mem.search_by_modality(q, k=1, modality="text")
        self.assertEqual(out.shape, (1, 4))
        self.assertEqual(meta[0]["modality"], "text")

if __name__ == "__main__":
    unittest.main()
