import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import tempfile

from src.hierarchical_memory import HierarchicalMemory


class TestHierarchicalMemory(unittest.TestCase):
    def test_add_and_search(self):
        torch.manual_seed(0)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        data = torch.randn(3, 4)
        mem.add(data, metadata=["a", "b", "c"])
        out, meta = mem.search(data[0], k=1)
        self.assertEqual(out.shape, (1, 4))
        self.assertEqual(len(meta), 1)
        self.assertIn(meta[0], ["a", "b", "c"])

    def test_save_and_load(self):
        torch.manual_seed(0)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=5)
        data = torch.randn(4, 4)
        mem.add(data, metadata=["a", "b", "c", "d"])
        expected, meta = mem.search(data[0], k=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            mem.save(tmpdir)
            loaded = HierarchicalMemory.load(tmpdir)
            out, meta2 = loaded.search(data[0], k=2)
        torch.testing.assert_close(out, expected)
        self.assertEqual(meta2, meta)


if __name__ == "__main__":
    unittest.main()
