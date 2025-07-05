import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.analogical_retrieval import analogy_offset, apply_analogy, analogy_search


class TestAnalogicalRetrieval(unittest.TestCase):
    def test_offset_and_apply(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        off = analogy_offset(a, b)
        torch.testing.assert_close(off, torch.tensor([-1.0, 1.0, 0.0]))
        q = torch.tensor([1.0, 0.0, 1.0])
        out = apply_analogy(q, off)
        torch.testing.assert_close(out, torch.tensor([0.0, 1.0, 1.0]))

    def test_memory_analogy_search(self):
        mem = HierarchicalMemory(dim=3, compressed_dim=3, capacity=10)
        # identity autoencoder for reproducibility
        mem.compressor.encoder.weight.data.copy_(torch.eye(3))
        mem.compressor.encoder.bias.data.zero_()
        mem.compressor.decoder.weight.data.copy_(torch.eye(3))
        mem.compressor.decoder.bias.data.zero_()

        vectors = {
            "man": torch.tensor([1.0, 0.0, 0.0]),
            "woman": torch.tensor([0.0, 1.0, 0.0]),
            "king": torch.tensor([1.0, 0.0, 1.0]),
            "queen": torch.tensor([0.0, 1.0, 1.0]),
        }
        for k, v in vectors.items():
            mem.add(v, metadata=[k])

        off = analogy_offset(vectors["man"], vectors["woman"])
        vec, meta = mem.search(vectors["king"], k=1, mode="analogy", offset=off)
        self.assertEqual(meta[0], "queen")

        # shortcut helper
        vec2, meta2 = analogy_search(mem, vectors["king"], vectors["man"], vectors["woman"], k=1)
        self.assertEqual(meta2[0], "queen")


if __name__ == "__main__":
    unittest.main()
