import unittest
import torch
from asi.transformer_circuits import (
    record_attention_weights,
    head_importance,
)


class TestTransformerCircuits(unittest.TestCase):
    def setUp(self):
        layer = torch.nn.TransformerEncoderLayer(d_model=8, nhead=2)
        self.model = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.input = torch.randn(4, 3, 8)  # seq, batch, dim

    def test_record_attention_weights(self):
        weights = record_attention_weights(self.model, self.input)
        self.assertTrue(weights)
        for w in weights.values():
            self.assertEqual(w.shape[-2:], (self.input.size(0), self.input.size(0)))

    def test_head_importance(self):
        imps = head_importance(self.model, self.input, 'layers.0.self_attn')
        self.assertEqual(imps.numel(), 2)
        self.assertTrue(torch.all(imps >= 0))


if __name__ == '__main__':
    unittest.main()
