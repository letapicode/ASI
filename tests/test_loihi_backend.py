import unittest
from unittest.mock import patch
import torch

import asi.spiking_layers as sl
import asi.loihi_backend as lb


class TestLoihiBackend(unittest.TestCase):
    def test_lif_offload_called(self):
        with patch.object(lb, '_HAS_LOIHI', True), \
             patch.object(lb, 'lif_forward', return_value=(torch.tensor([1.0]), torch.tensor([0.0]))) as lf:
            neuron = sl.LIFNeuron(use_loihi=True)
            x = torch.tensor([0.5])
            init_mem = neuron.mem.clone()
            out = neuron(x)
            lf.assert_called_with(init_mem, x, neuron.decay, neuron.threshold)
            self.assertTrue(torch.allclose(out, torch.tensor([1.0])))

    def test_linear_offload_called(self):
        with patch.object(lb, '_HAS_LOIHI', True), \
             patch.object(lb, 'linear_forward', return_value=torch.ones(1, 2)) as lin, \
             patch.object(lb, 'lif_forward', return_value=(torch.zeros(1, 2), torch.zeros(1, 2))):
            layer = sl.SpikingLinear(2, 2, use_loihi=True)
            x = torch.randn(1, 2)
            layer(x)
            lin.assert_called_with(x, layer.linear.weight, layer.linear.bias)

    def test_fallback_cpu(self):
        torch.manual_seed(0)
        neuron1 = sl.LIFNeuron()
        neuron2 = sl.LIFNeuron(use_loihi=True)
        x = torch.randn(2, 3)
        out1 = neuron1(x)
        out2 = neuron2(x)
        self.assertTrue(torch.allclose(out1, out2))


if __name__ == '__main__':
    unittest.main()

