import unittest
import torch

from asi.cross_modal_fusion import CrossModalFusion


class TestCrossModalFusion(unittest.TestCase):
    def test_fusion_output(self):
        model = CrossModalFusion(vocab_size=10, audio_dim=4, image_dim=4, hidden=8)
        text = torch.randint(0, 10, (2, 3))
        image = torch.randn(2, 3, 4)
        audio = torch.randn(2, 3, 4)
        out = model(text, image, audio)
        self.assertEqual(out.shape, (2, 3, 8))


if __name__ == '__main__':
    unittest.main()
