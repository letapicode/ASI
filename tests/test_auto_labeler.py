import unittest
import numpy as np
import torch

from asi.auto_labeler import AutoLabeler, AutoLabelerConfig
from asi.multimodal_world_model import MultiModalWorldModel, MultiModalWorldModelConfig


class TestAutoLabeler(unittest.TestCase):
    def test_label(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2, embed_dim=4)
        model = MultiModalWorldModel(cfg)
        labeler = AutoLabeler(model, AutoLabelerConfig(num_labels=3))
        texts = ["aa", "bb"]
        images = [np.zeros((1, 4, 4), dtype=np.float32) for _ in texts]
        actions = [0, 1]
        labels = labeler.label(zip(texts, images, actions))
        self.assertEqual(len(labels), 2)
        for l in labels:
            self.assertTrue(0 <= l < 3)


if __name__ == "__main__":
    unittest.main()
