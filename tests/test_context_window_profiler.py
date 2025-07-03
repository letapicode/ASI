import unittest
import torch
from asi.context_profiler import ContextWindowProfiler

class TestContextWindowProfiler(unittest.TestCase):
    def test_run(self):
        model = torch.nn.Linear(8, 4)
        profiler = ContextWindowProfiler(model, [2, 4])
        results = profiler.run()
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("cpu_time", r)
            self.assertIn("gpu_mem", r)

if __name__ == "__main__":
    unittest.main()
