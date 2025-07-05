import unittest
import importlib.machinery
import importlib.util
import sys
import torch

loader = importlib.machinery.SourceFileLoader('sl', 'src/spiking_layers.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
sl = importlib.util.module_from_spec(spec)
sys.modules['sl'] = sl
sl.__package__ = 'src'
loader.exec_module(sl)
LIFNeuron = sl.LIFNeuron
SpikingLinear = sl.SpikingLinear


class TestSpikingLayers(unittest.TestCase):
    def test_forward_shape(self):
        layer = SpikingLinear(4, 3)
        x = torch.randn(2, 4)
        out = layer(x)
        self.assertEqual(out.shape, (2, 3))

    def test_gradient_flow(self):
        layer = SpikingLinear(5, 2)
        x = torch.randn(4, 5, requires_grad=True)
        out = layer(x).sum()
        out.backward()
        self.assertIsNotNone(layer.linear.weight.grad)
        self.assertFalse(torch.all(layer.linear.weight.grad == 0))


if __name__ == "__main__":
    unittest.main()
