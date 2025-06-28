import tempfile
import unittest
import asyncio
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
        self.assertEqual(out.device, data[0].device)
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

    def test_faiss_backend(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, db_path=tmpdir)
            data = torch.randn(3, 4)
            mem.add(data, metadata=["x", "y", "z"])
            # ensure vectors are saved by adding and reloading
            mem2 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, db_path=tmpdir)
            out, meta = mem2.search(data[0], k=1)
            self.assertEqual(len(meta), 1)

    def test_async_add_search(self):
        torch.manual_seed(0)
        async def run():
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, use_async=True)
            data = torch.randn(2, 4)
            await mem.aadd(data, metadata=["a", "b"])
            out, meta = await mem.asearch(data[0], k=1)
            self.assertEqual(out.shape, (1, 4))
            self.assertEqual(len(meta), 1)
            self.assertIn(meta[0], ["a", "b"])
        asyncio.run(run())

    def test_async_save_load(self):
        torch.manual_seed(0)
        async def run():
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, use_async=True)
            data = torch.randn(3, 4)
            await mem.aadd(data, metadata=["x", "y", "z"])
            out_before, meta_before = await mem.asearch(data[0], k=1)
            with tempfile.TemporaryDirectory() as tmpdir:
                await mem.save_async(tmpdir)
                loaded = await HierarchicalMemory.load_async(tmpdir, use_async=True)
                out_after, meta_after = await loaded.asearch(data[0], k=1)
            torch.testing.assert_close(out_after, out_before)
            self.assertEqual(meta_after, meta_before)
        asyncio.run(run())

    def test_delete(self):
        torch.manual_seed(0)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        data = torch.randn(3, 4)
        mem.add(data, metadata=["a", "b", "c"])
        mem.delete(tag="b")
        self.assertEqual(len(mem.store), 2)

    def test_delete_persist_faiss(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, db_path=tmpdir)
            data = torch.randn(3, 4)
            mem.add(data, metadata=["x", "y", "z"])
            mem.delete(index=1)
            self.assertEqual(len(mem.store), 2)
            mem2 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, db_path=tmpdir)
            self.assertEqual(len(mem2.store), 2)


if __name__ == "__main__":
    unittest.main()
