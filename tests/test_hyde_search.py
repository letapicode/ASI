import unittest
import numpy as np
import torch

from asi.vector_store import VectorStore
from asi.hierarchical_memory import HierarchicalMemory


class TestHyDESearch(unittest.TestCase):
    def test_vector_store_hyde_fallback(self):
        store = VectorStore(dim=3)
        vecs = np.eye(3, dtype=np.float32)
        store.add(vecs, metadata=["a", "b", "c"])
        q = vecs[0]
        v1, m1 = store.search(q, k=1)
        v2, m2 = store.hyde_search(q, k=1)
        np.testing.assert_allclose(v2, v1)
        self.assertEqual(m2, m1)

    def test_hierarchical_memory_hyde_mode(self):
        torch.manual_seed(0)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        data = torch.randn(2, 4)
        mem.add(data, metadata=["x", "y"])
        q = data[0]
        v1, m1 = mem.search(q, k=1)
        v2, m2 = mem.search(q, k=1, mode="hyde")
        self.assertEqual(m2, m1)
        torch.testing.assert_close(v2, v1)

    def test_async_hyde_mode(self):
        torch.manual_seed(0)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, use_async=True)
        data = torch.randn(2, 4)
        mem.add(data, metadata=["x", "y"])
        q = data[0]
        import asyncio

        async def run():
            v1, m1 = await mem.asearch(q, k=1)
            v2, m2 = await mem.asearch(q, k=1, mode="hyde")
            self.assertEqual(m2, m1)
            torch.testing.assert_close(v2, v1)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
