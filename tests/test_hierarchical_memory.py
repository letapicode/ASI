import tempfile
import unittest
import asyncio
import torch

from asi.hierarchical_memory import (
    HierarchicalMemory,
    MemoryServer,
    push_remote,
    query_remote,
    push_remote_async,
    query_remote_async,
    push_batch_remote,
    query_batch_remote,
    push_batch_remote_async,
    query_batch_remote_async,
)
try:
    import grpc  # noqa: F401
    _HAS_GRPC = True
except Exception:
    _HAS_GRPC = False


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
        self.assertEqual(len(mem), 2)

    def test_delete_persist_faiss(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, db_path=tmpdir)
            data = torch.randn(3, 4)
            mem.add(data, metadata=["x", "y", "z"])
            mem.delete(index=1)
            self.assertEqual(len(mem), 2)
            mem2 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, db_path=tmpdir)
            self.assertEqual(len(mem2), 2)

    def test_modalities(self):
        torch.manual_seed(0)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        text = torch.randn(2, 4)
        images = torch.randn(2, 4)
        mem.add_modalities(text, images, metadata=["a", "b"])
        q = text[0]
        out, meta = mem.search_by_modality(q, k=1, modality="text")
        self.assertEqual(out.shape, (1, 4))
        self.assertEqual(meta[0]["modality"], "text")

    def test_sync_methods_inside_event_loop(self):
        torch.manual_seed(0)

        async def run():
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, use_async=True)
            data = torch.randn(2, 4)

            t = mem.add(data, metadata=["a", "b"])
            if isinstance(t, asyncio.Task):
                await t

            t = mem.search(data[0], k=1)
            if isinstance(t, asyncio.Task):
                out, meta = await t
            else:
                out, meta = t
            self.assertEqual(out.shape, (1, 4))
            self.assertEqual(len(meta), 1)

            t = mem.delete(tag="a")
            if isinstance(t, asyncio.Task):
                await t

            with tempfile.TemporaryDirectory() as tmpdir:
                t = mem.save(tmpdir)
                if isinstance(t, asyncio.Task):
                    await t
                loaded = HierarchicalMemory.load(tmpdir, use_async=True)
                if isinstance(loaded, asyncio.Task):
                    loaded = await loaded
                self.assertIsInstance(loaded, HierarchicalMemory)

        asyncio.run(run())

    def test_grpc_server(self):
        if not _HAS_GRPC:
            self.skipTest("grpcio not available")

        torch.manual_seed(0)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        server = MemoryServer(mem, address="localhost:50070", max_workers=1)
        server.start()

        data = torch.randn(1, 4)
        ok = push_remote("localhost:50070", data[0], metadata="r")
        self.assertTrue(ok)
        vec, meta = query_remote("localhost:50070", data[0], k=1)
        self.assertEqual(vec.shape, (1, 4))
        self.assertEqual(meta[0], "r")
        server.stop(0)

    def test_grpc_server_batch(self):
        if not _HAS_GRPC:
            self.skipTest("grpcio not available")

        torch.manual_seed(0)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        server = MemoryServer(mem, address="localhost:50072", max_workers=1)
        server.start()

        data = torch.randn(2, 4)
        ok = push_batch_remote("localhost:50072", data, metadata=["x", "y"])
        self.assertTrue(ok)
        vec, meta = query_batch_remote("localhost:50072", data, k=1)
        self.assertEqual(vec.shape, (2, 1, 4))
        self.assertEqual(len(meta), 2)
        server.stop(0)

    def test_grpc_server_async(self):
        if not _HAS_GRPC:
            self.skipTest("grpcio not available")

        torch.manual_seed(0)

        async def run() -> None:
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
            server = MemoryServer(mem, address="localhost:50071", max_workers=1)
            server.start()
            data = torch.randn(1, 4)
            ok = await push_remote_async("localhost:50071", data[0], metadata="r")
            self.assertTrue(ok)
            vec, meta = await query_remote_async("localhost:50071", data[0], k=1)
            self.assertEqual(vec.shape, (1, 4))
            self.assertEqual(meta[0], "r")
            server.stop(0)

        asyncio.run(run())

    def test_grpc_server_async_batch(self):
        if not _HAS_GRPC:
            self.skipTest("grpcio not available")

        torch.manual_seed(0)

        async def run() -> None:
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
            server = MemoryServer(mem, address="localhost:50073", max_workers=1)
            server.start()
            data = torch.randn(2, 4)
            ok = await push_batch_remote_async(
                "localhost:50073", data, metadata=["m", "n"]
            )
            self.assertTrue(ok)
            vec, meta = await query_batch_remote_async(
                "localhost:50073", data, k=1
            )
            self.assertEqual(vec.shape, (2, 1, 4))
            self.assertEqual(len(meta), 2)
            server.stop(0)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
