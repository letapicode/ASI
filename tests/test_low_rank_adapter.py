import unittest
import torch
from torch import nn

from asi.low_rank_adapter import LowRankLinear, apply_low_rank_adaptation


class Dummy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)


class TestLowRankAdapter(unittest.TestCase):
    def test_apply_and_forward(self):
        model = Dummy()
        apply_low_rank_adaptation(model, ["fc"], r=2, alpha=1.0)
        self.assertIsInstance(model.fc, LowRankLinear)
        x = torch.randn(3, 4)
        y = model.fc(x)
        self.assertEqual(y.shape, (3, 2))


if __name__ == "__main__":  # pragma: no cover - CLI
    unittest.main()
