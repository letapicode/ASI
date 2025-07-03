import unittest
import torch
import numpy as np

from asi.sensorimotor_pretrainer import (
    SensorimotorPretrainConfig,
    SensorimotorLogDataset,
    pretrain_sensorimotor,
)
from asi.multimodal_world_model import MultiModalWorldModel, MultiModalWorldModelConfig


def tokenizer(t: str):
    return [ord(c) % 10 for c in t]


class TestSensorimotorPretrainer(unittest.TestCase):
    def test_pretrain_runs(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2)
        model = MultiModalWorldModel(cfg)
        data = []
        for _ in range(4):
            t = "hi"
            img = np.zeros((1, 8, 8), dtype=np.float32)
            a = 1
            nt = "hi"
            nimg = np.ones((1, 8, 8), dtype=np.float32)
            data.append((t, img, a, nt, nimg))
        dataset = SensorimotorLogDataset(data, tokenizer)
        pcfg = SensorimotorPretrainConfig(epochs=1, batch_size=2)
        out = pretrain_sensorimotor(model, dataset, pcfg)
        self.assertIsInstance(out, MultiModalWorldModel)


if __name__ == "__main__":
    unittest.main()
