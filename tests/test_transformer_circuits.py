import unittest
from pathlib import Path
import torch
from asi.transformer_circuits import (
    record_attention_weights,
    head_importance,
    AttentionVisualizer,
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

    def test_attention_visualizer(self):
        out = Path('tmp_vis')
        vis = AttentionVisualizer(self.model, ['layers.0.self_attn'], out_dir=str(out))
        weights = vis.run(self.input)
        self.assertIn('layers.0.self_attn', weights)
        self.assertTrue(any('layers_0_self_attn_h' in p.name for p in out.glob('*.png')))
        for p in out.glob('*'):
            p.unlink()
        out.rmdir()


if __name__ == '__main__':
    unittest.main()
