import unittest
import torch

from asi.cross_modal_fusion import (
    CrossModalFusionConfig,
    CrossModalFusion,
    MultiModalDataset,
    train_fusion_model,
    encode_all,
)


def simple_tokenizer(text: str):
    return [ord(c) % 50 for c in text]


class TestCrossModalFusion(unittest.TestCase):
    def setUp(self):
        cfg = CrossModalFusionConfig(
            vocab_size=50,
            text_dim=16,
            img_channels=3,
            audio_channels=1,
            latent_dim=8,
            temperature=0.1,
        )
        self.model = CrossModalFusion(cfg)
        img = torch.randn(3, 8, 8)
        aud = torch.randn(1, 16)
        self.dataset = MultiModalDataset(
            [
                ("a", img, aud),
                ("b", img.clone(), aud.clone()),
            ],
            simple_tokenizer,
        )

    def test_forward_shapes(self):
        tokens, img, aud = self.dataset[0]
        tokens = tokens.unsqueeze(0)
        img = img.unsqueeze(0)
        aud = aud.unsqueeze(0)
        t, i, a = self.model(tokens, img, aud)
        self.assertEqual(t.shape, (1, 8))
        self.assertEqual(i.shape, (1, 8))
        self.assertEqual(a.shape, (1, 8))

    def test_encode_all(self):
        t_vecs, i_vecs, a_vecs = encode_all(self.model, self.dataset, batch_size=1)
        self.assertEqual(t_vecs.shape, (2, 8))
        self.assertEqual(i_vecs.shape, (2, 8))
        self.assertEqual(a_vecs.shape, (2, 8))

    def test_train_updates_params(self):
        before = self.model.text_proj.fc.weight.clone()
        train_fusion_model(self.model, self.dataset, epochs=1, batch_size=1)
        after = self.model.text_proj.fc.weight
        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
