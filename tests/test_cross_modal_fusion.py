import unittest
import torch
from asi.cross_modal_fusion import CrossModalFusion, train_fusion_step


class TestCrossModalFusion(unittest.TestCase):
    def test_encode_and_train(self):
        model = CrossModalFusion(vocab=10)
        text = torch.randint(0, 10, (2, 4))
        images = torch.randn(2, 3, 8, 8)
        audio = torch.randn(2, 80, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = train_fusion_step(model, text, images, audio, opt)
        self.assertIsInstance(loss, float)
        emb = model.encode_text(text)
        self.assertEqual(emb.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
