import unittest
import torch
from torch import nn

from asi.formal_verifier import verify_model


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class TestFormalVerifier(unittest.TestCase):
    def test_verify(self):
        model = SimpleModel()
        inputs = [torch.randn(1, 4) for _ in range(3)]
        self.assertTrue(verify_model(model, inputs))


if __name__ == '__main__':
    unittest.main()
