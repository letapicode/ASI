import unittest
import torch

from asi.multimodal_world_model import MultiModalWorldModel, MultiModalWorldModelConfig, TrajectoryDataset
from asi.world_model_distiller import DistillConfig, distill_world_model


class TestWorldModelDistiller(unittest.TestCase):
    def test_distillation_runs(self):
        cfg_t = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2)
        cfg_s = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2, embed_dim=32)
        teacher = MultiModalWorldModel(cfg_t)
        student = MultiModalWorldModel(cfg_s)

        data = []
        for _ in range(4):
            t = torch.randint(0, 10, (1, 4))
            img = torch.randn(1, 1, 8, 8)
            a = torch.randint(0, 2, (1,))
            nt = torch.randint(0, 10, (1, 4))
            nimg = torch.randn(1, 1, 8, 8)
            r = torch.randn(())
            data.append((t, img, a, nt, nimg, r))
        dataset = TrajectoryDataset(data, lambda x: [int(c) for c in x[0]])
        cfg = DistillConfig(epochs=1, batch_size=2)
        out = distill_world_model(teacher, student, dataset, cfg)
        self.assertIsInstance(out, MultiModalWorldModel)


if __name__ == "__main__":
    unittest.main()
