import unittest
import torch

from asi.gradient_compression import GradientCompressionConfig, GradientCompressor


class TestGradientCompressor(unittest.TestCase):
    def test_topk(self):
        cfg = GradientCompressionConfig(topk=2)
        comp = GradientCompressor(cfg)
        g = torch.tensor([1.0, -3.0, 2.0, 0.5])
        out = comp.compress(g.clone())
        self.assertEqual(int((out != 0).sum()), 2)
        # Ensure top magnitudes preserved
        idx = torch.topk(g.abs(), 2).indices
        self.assertTrue(torch.all(out[idx] != 0))

    def test_quantize(self):
        g = torch.tensor([0.1, -0.2, 0.3, -0.4])
        cfg = GradientCompressionConfig(bits=4)
        comp = GradientCompressor(cfg)
        out = comp.compress(g.clone())
        self.assertTrue(torch.allclose(out, g, atol=0.1))

    def test_combined(self):
        g = torch.tensor([0.6, -0.3, 0.2, 0.1])
        cfg = GradientCompressionConfig(topk=1, bits=3)
        comp = GradientCompressor(cfg)
        out = comp.compress(g.clone())
        self.assertEqual(int((out != 0).sum()), 1)
        self.assertTrue(torch.allclose(out[out != 0], torch.tensor([0.6]), atol=0.2))


if __name__ == "__main__":
    unittest.main()
