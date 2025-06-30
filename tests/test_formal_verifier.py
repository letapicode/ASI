import unittest
import torch
from torch import nn

from asi.formal_verifier import verify_model, weight_bound


class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 2)

    def forward(self, x):
        return self.w(x)


class TestFormalVerifier(unittest.TestCase):
    def test_verify(self):
        model = Simple()
        checks = [weight_bound(10.0)]
        self.assertTrue(verify_model(model, checks))
        with torch.no_grad():
            model.w.weight.mul_(1000)
        self.assertFalse(verify_model(model, checks))


if __name__ == "__main__":
    unittest.main()
