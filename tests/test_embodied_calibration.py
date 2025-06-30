import unittest
import torch
from torch import nn

from asi.embodied_calibration import calibrate


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(2, 2)

    def forward(self, x):
        return self.l(x)


class TestEmbodiedCalibration(unittest.TestCase):
    def test_calibrate(self):
        model = DummyModel()
        sim = [(torch.randn(1, 2), torch.randn(1, 2)) for _ in range(2)]
        real = [(torch.randn(1, 2), torch.randn(1, 2)) for _ in range(2)]
        calibrate(model, sim, real, epochs=1)


if __name__ == "__main__":
    unittest.main()
