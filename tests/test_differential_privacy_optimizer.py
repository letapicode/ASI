import unittest
import torch
from asi.differential_privacy_optimizer import DifferentialPrivacyConfig, DifferentialPrivacyOptimizer

class TestDifferentialPrivacyOptimizer(unittest.TestCase):
    def test_clip_and_noise(self):
        p = torch.nn.Parameter(torch.zeros(1))
        cfg = DifferentialPrivacyConfig(lr=1.0, clip_norm=0.5, noise_std=0.0)
        opt = DifferentialPrivacyOptimizer([p], cfg)
        p.grad = torch.tensor([10.0])
        opt.step()
        self.assertLessEqual(float(p.grad.abs()), 0.5)

    def test_noise_changes_param(self):
        p = torch.nn.Parameter(torch.zeros(1))
        cfg = DifferentialPrivacyConfig(lr=0.0, clip_norm=1.0, noise_std=0.1)
        opt = DifferentialPrivacyOptimizer([p], cfg)
        p.grad = torch.tensor([0.0])
        opt.step()
        self.assertNotEqual(float(p.grad), 0.0)

if __name__ == "__main__":
    unittest.main()
