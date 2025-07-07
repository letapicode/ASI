import importlib.machinery
import importlib.util
import sys
import unittest
import torch

loader_ds = importlib.machinery.SourceFileLoader('bci_ds', 'src/bci_dataset.py')
spec_ds = importlib.util.spec_from_loader(loader_ds.name, loader_ds)
bci_ds = importlib.util.module_from_spec(spec_ds)
loader_ds.exec_module(bci_ds)
BCIDataset = bci_ds.BCIDataset
load_synthetic_bci = bci_ds.load_synthetic_bci

loader_cmf = importlib.machinery.SourceFileLoader('cmf', 'src/cross_modal_fusion.py')
spec_cmf = importlib.util.spec_from_loader(loader_cmf.name, loader_cmf)
cmf = importlib.util.module_from_spec(spec_cmf)
loader_cmf.exec_module(cmf)
CrossModalFusionConfig = cmf.CrossModalFusionConfig
CrossModalFusion = cmf.CrossModalFusion
MultiModalDataset = cmf.MultiModalDataset
encode_all = cmf.encode_all

from asi.hierarchical_memory import HierarchicalMemory


def simple_tokenizer(text: str):
    return [ord(c) % 50 for c in text]


class TestBCIDataset(unittest.TestCase):
    def test_synthetic_retrieval(self):
        bci = load_synthetic_bci(num_samples=2, channels=2, length=32)
        texts = ["x", "y"]
        imgs = [torch.randn(3, 16, 16) for _ in texts]
        auds = [torch.randn(1, 32) for _ in texts]
        samples = [(texts[i], imgs[i], auds[i], bci[i]) for i in range(2)]
        ds = MultiModalDataset(samples, simple_tokenizer, bci_shape=(2, 32))
        cfg = CrossModalFusionConfig(vocab_size=50, text_dim=8, img_channels=3, audio_channels=1, bci_channels=2, latent_dim=4)
        model = CrossModalFusion(cfg)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        t, i, a, b = encode_all(model, ds, batch_size=1, memory=mem, include_bci=True)
        q = (t[0] + i[0] + a[0] + b[0]) / 4.0
        out, meta = mem.search(q, k=1)
        self.assertEqual(meta[0], 0)


if __name__ == '__main__':
    unittest.main()
