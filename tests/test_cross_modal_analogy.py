import unittest
import importlib.machinery
import importlib.util
import sys
import torch

# load modules dynamically from src
loader_fusion = importlib.machinery.SourceFileLoader('cmf', 'src/cross_modal_fusion.py')
spec_fusion = importlib.util.spec_from_loader(loader_fusion.name, loader_fusion)
cmf = importlib.util.module_from_spec(spec_fusion)
loader_fusion.exec_module(cmf)
sys.modules['asi.cross_modal_fusion'] = cmf
CrossModalFusionConfig = cmf.CrossModalFusionConfig
CrossModalFusion = cmf.CrossModalFusion
MultiModalDataset = cmf.MultiModalDataset

loader_analogy = importlib.machinery.SourceFileLoader('cma', 'src/cross_modal_analogy.py')
spec_analogy = importlib.util.spec_from_loader(loader_analogy.name, loader_analogy)
cma = importlib.util.module_from_spec(spec_analogy)
loader_analogy.exec_module(cma)
sys.modules['asi.cross_modal_analogy'] = cma
cross_modal_analogy_search = cma.cross_modal_analogy_search

from asi.hierarchical_memory import HierarchicalMemory


def dummy_tokenizer(text: str):
    return [ord(c) % 50 for c in text]


class TestCrossModalAnalogy(unittest.TestCase):
    def test_search(self):
        torch.manual_seed(0)
        cfg = CrossModalFusionConfig(vocab_size=50, text_dim=4, img_channels=3, audio_channels=1, latent_dim=4)
        model = CrossModalFusion(cfg)
        # identity autoencoder for deterministic behavior
        mem = HierarchicalMemory(dim=4, compressed_dim=4, capacity=10)
        mem.compressor.encoder.weight.data.copy_(torch.eye(4))
        mem.compressor.encoder.bias.data.zero_()
        mem.compressor.decoder.weight.data.copy_(torch.eye(4))
        mem.compressor.decoder.bias.data.zero_()

        # patch encode_all to use simple embeddings
        base = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ])

        def fake_encode_all(model, dataset, batch_size=8, memory=None, **kw):
            t = base
            i = torch.zeros_like(base)
            a = torch.zeros_like(base)
            if memory is not None:
                memory.add_multimodal(t, i, a, metadata=list(range(len(base))))
            return t, i, a

        cmf.encode_all = fake_encode_all
        texts = ["a", "b", "c", "d"]
        imgs = [torch.zeros(3, 1, 1) for _ in texts]
        auds = [torch.zeros(1, 1) for _ in texts]
        ds = MultiModalDataset(list(zip(texts, imgs, auds)), dummy_tokenizer)
        vec, meta = cross_modal_analogy_search(model, ds, mem, 2, 0, 1, k=1, batch_size=1)
        self.assertEqual(meta[0], 3)


if __name__ == '__main__':
    unittest.main()
