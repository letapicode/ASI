import unittest
import torch
from asi.context_profiler import ContextWindowProfiler

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(100, 4)
        self.fc = torch.nn.Linear(4, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.emb(x)).mean(dim=1)

class TestContextWindowProfiler(unittest.TestCase):
    def test_profile(self):
        model = ToyModel()
        profiler = ContextWindowProfiler(model)
        stats = profiler.profile([4, 8])
        self.assertEqual(len(stats), 2)
        for s in stats:
            self.assertIn('cpu_time', s)
            self.assertIn('gpu_mem', s)

if __name__ == '__main__':
    unittest.main()
