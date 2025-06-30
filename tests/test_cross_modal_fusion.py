import unittest
from unittest.mock import patch
import torch

from asi.cross_modal_fusion import (
    CrossModalFusionConfig,
    CrossModalFusion,
    MultiModalDataset,
    train_fusion_model,
)


class TestCrossModalFusion(unittest.TestCase):
    def test_train_uses_cfg_lr(self):
        cfg = CrossModalFusionConfig(
            vocab_size=10,
            text_dim=8,
            img_channels=1,
            audio_channels=1,
            latent_dim=2,
            lr=0.123,
        )
        model = CrossModalFusion(cfg)

        def tokenizer(x):
            return [0, 1]

        dataset = MultiModalDataset(
            [("a", torch.zeros(1, 32, 32), torch.zeros(1, 16))], tokenizer
        )

        with patch("torch.optim.Adam") as AdamMock:
            opt = AdamMock.return_value
            train_fusion_model(model, dataset, epochs=1, batch_size=1)
            self.assertEqual(AdamMock.call_args.kwargs["lr"], cfg.lr)
            assert opt.step.called


if __name__ == "__main__":
    unittest.main()
