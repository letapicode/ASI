import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.link_slot_attention import LinkSlotAttention


class TestLinkSlotAttention(unittest.TestCase):
    def test_forward_shape(self):
        torch.manual_seed(0)
        memory = HierarchicalMemory(dim=4, compressed_dim=2, capacity=5)
        module = LinkSlotAttention(memory, dim=4, k_top=2)
        x = torch.randn(1, 3, 4)
        out = module(x)
        self.assertEqual(out.shape, x.shape)

    def test_memory_growth(self):
        torch.manual_seed(0)
        memory = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        module = LinkSlotAttention(memory, dim=4, k_top=1)
        x = torch.randn(1, 2, 4)
        module(x)
        self.assertGreaterEqual(len(memory.store), 2)


if __name__ == "__main__":
    unittest.main()
