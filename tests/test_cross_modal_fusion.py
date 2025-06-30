import unittest
import torch

from asi.cross_modal_fusion import CrossModalFusionModel, train_fusion_model


class TestCrossModalFusion(unittest.TestCase):
    def test_forward_shapes(self):
        model = CrossModalFusionModel(vocab_size=10, dim=16)
        text = torch.randint(0, 10, (2, 5))
        images = torch.randn(2, 3, 32, 32)
        audio = torch.randn(2, 1, 64)
        reps = model(text=text, images=images, audio=audio)
        self.assertEqual(reps["text"].shape, (2, 16))
        self.assertEqual(reps["image"].shape, (2, 16))
        self.assertEqual(reps["audio"].shape, (2, 16))

    def test_train_step(self):
        model = CrossModalFusionModel(vocab_size=10, dim=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def loader():
            for _ in range(2):
                yield {
                    "text": torch.randint(0, 10, (2, 4)),
                    "images": torch.randn(2, 3, 16, 16),
                    "audio": torch.randn(2, 1, 32),
                }

        train_fusion_model(loader(), model, opt, steps=2)


if __name__ == "__main__":
    unittest.main()
