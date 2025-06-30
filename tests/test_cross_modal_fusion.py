import unittest
import torch
from asi.cross_modal_fusion import (
    CrossModalFusionConfig,
    CrossModalFusion,
    MultiModalDataset,
    train_fusion_model,
    encode_all,
)


def _tokenizer(text: str):
    ids = [(ord(c) - 97) % 20 + 1 for c in text]
    ids = (ids + [0] * 4)[:4]
    return ids


def _make_dataset(n=4):
    triples = []
    for i in range(n):
        text = "test"[i % 4 :] + "a" * (i % 2)
        img = torch.randn(3, 16, 16)
        aud = torch.randn(1, 32)
        triples.append((text, img, aud))
    return MultiModalDataset(triples, _tokenizer)


class TestCrossModalFusion(unittest.TestCase):
    def test_forward_shapes(self):
        cfg = CrossModalFusionConfig(
            vocab_size=32,
            text_dim=16,
            img_channels=3,
            audio_channels=1,
            latent_dim=8,
        )
        model = CrossModalFusion(cfg)
        ds = _make_dataset(1)
        tokens, img, aud = ds[0]
        t, i, a = model(
            tokens.unsqueeze(0), img.unsqueeze(0), aud.unsqueeze(0)
        )
        self.assertEqual(t.shape, (1, 8))
        self.assertEqual(i.shape, (1, 8))
        self.assertEqual(a.shape, (1, 8))

    def test_train_and_encode(self):
        cfg = CrossModalFusionConfig(
            vocab_size=32,
            text_dim=16,
            img_channels=3,
            audio_channels=1,
            latent_dim=8,
        )
        ds = _make_dataset(6)
        model = CrossModalFusion(cfg)
        before = model.text_enc.embed.weight.clone()
        train_fusion_model(model, ds, epochs=1, batch_size=2)
        self.assertFalse(torch.allclose(before, model.text_enc.embed.weight))
        t_vecs, i_vecs, a_vecs = encode_all(model, ds, batch_size=2)
        self.assertEqual(t_vecs.shape[0], len(ds))
        self.assertEqual(i_vecs.shape, t_vecs.shape)
        self.assertEqual(a_vecs.shape, t_vecs.shape)


if __name__ == "__main__":
    unittest.main()
