import unittest
import torch

from asi.multimodal_world_model import MultiModalWorldModel


class TestMultiModalWorldModel(unittest.TestCase):
    def test_forward(self):
        model = MultiModalWorldModel(vocab_size=10, image_dim=4, action_dim=5, hidden=8)
        text = torch.randint(0, 10, (2, 3))
        image = torch.randn(2, 3, 4)
        action = torch.randint(0, 5, (2, 3))
        out = model(text, image, action)
        self.assertEqual(out.shape, (2, 3, 8))


if __name__ == '__main__':
    unittest.main()
