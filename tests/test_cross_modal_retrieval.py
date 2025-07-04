import importlib.machinery
import importlib.util
import sys
import unittest
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
retrieval_accuracy = cmf.retrieval_accuracy

from asi.hierarchical_memory import HierarchicalMemory


def simple_tokenizer(text: str):
    return [ord(c) % 50 for c in text]


class TestCrossModalRetrieval(unittest.TestCase):
    def test_accuracy(self):
        cfg = CrossModalFusionConfig(
            vocab_size=50,
            text_dim=8,
            img_channels=3,
            audio_channels=1,
            latent_dim=4,
        )
        model = CrossModalFusion(cfg)
        triples = [
            ("aa", torch.randn(3, 16, 16), torch.randn(1, 32)),
            ("bb", torch.randn(3, 16, 16), torch.randn(1, 32)),
        ]
        dataset = MultiModalDataset(triples, simple_tokenizer)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        acc = retrieval_accuracy(model, dataset, mem, batch_size=1, k=1)
        self.assertGreaterEqual(acc, 0.5)


if __name__ == '__main__':
    unittest.main()
