import unittest
import torch
from asi.hierarchical_memory import HierarchicalMemory

class TestAdaptiveEviction(unittest.TestCase):
    def test_eviction(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10, evict_limit=5)
        data = torch.randn(6, 2)
        mem.add(data)
        self.assertLessEqual(len(mem), 5)

if __name__ == '__main__':
    unittest.main()
