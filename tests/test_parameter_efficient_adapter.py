import unittest
import torch
from torch import nn

from asi.parameter_efficient_adapter import ParameterEfficientAdapter, PEFTConfig


class DummyData:
    def __iter__(self):
        for _ in range(10):
            x = torch.randn(2, 4)
            y = torch.randn(2, 1)
            yield x, y


class TestParameterEfficientAdapter(unittest.TestCase):
    def test_fit(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        cfg = PEFTConfig(target_modules=["0", "2"], r=2, lr=1e-2, epochs=2)
        adapter = ParameterEfficientAdapter(model, cfg)
        before = float(model(torch.zeros(1, 4)).abs().mean())
        adapter.fit(DummyData(), nn.MSELoss())
        after = float(model(torch.zeros(1, 4)).abs().mean())
        self.assertTrue(after != before)


if __name__ == "__main__":  # pragma: no cover - CLI
    unittest.main()
