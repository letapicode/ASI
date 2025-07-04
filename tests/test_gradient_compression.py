import importlib.machinery
import importlib.util
import sys
import unittest
import torch

loader = importlib.machinery.SourceFileLoader('asi.gradient_compression', 'src/gradient_compression.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
sys.modules['asi.gradient_compression'] = mod
loader.exec_module(mod)
GradientCompressionConfig = mod.GradientCompressionConfig
GradientCompressor = mod.GradientCompressor

class TestGradientCompressor(unittest.TestCase):
    def test_topk(self):
        cfg = GradientCompressionConfig(topk=2)
        comp = GradientCompressor(cfg)
        grad = torch.tensor([1.0, 0.5, 0.2])
        out = comp.compress(grad)
        self.assertEqual(int((out != 0).sum()), 2)

    def test_quantize(self):
        cfg = GradientCompressionConfig(bits=4)
        comp = GradientCompressor(cfg)
        grad = torch.linspace(-1, 1, 5)
        out = comp.compress(grad)
        self.assertTrue(torch.all(out.abs() <= 1.0))

    def test_dict(self):
        cfg = GradientCompressionConfig(topk=1, bits=8)
        comp = GradientCompressor(cfg)
        grads = {'a': torch.tensor([0.1, 0.9]), 'b': torch.tensor([1.5, -0.5])}
        out = comp.compress_dict(grads)
        self.assertEqual(set(out.keys()), {'a', 'b'})
        self.assertEqual(int((out['a'] != 0).sum()), 1)

if __name__ == '__main__':
    unittest.main()

