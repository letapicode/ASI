import tempfile
import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory


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
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        data = torch.randn(3, 4)
        mem.add(data, metadata=["x", "y", "z"])
        out_before, meta_before = mem.search(data[0], k=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mem.save(tmpdir)
            loaded = HierarchicalMemory.load(tmpdir)
            out_after, meta_after = loaded.search(data[0], k=1)
        torch.testing.assert_close(out_after, out_before)
        self.assertEqual(meta_after, meta_before)


if __name__ == "__main__":
    unittest.main()
