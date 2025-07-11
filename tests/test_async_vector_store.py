import os
import tempfile
import unittest
import asyncio
import numpy as np

from asi.vector_stores import AsyncFaissVectorStore


class TestAsyncFaissVectorStore(unittest.TestCase):
    def test_async_add_and_search(self):
        store = AsyncFaissVectorStore(dim=2)
        store.add_async(np.array([[1.0, 0.0], [0.0, 1.0]]), metadata=["a", "b"]).result()
        future = store.search_async(np.array([0.0, 1.0]), k=1)
        vecs, meta = future.result()
        np.testing.assert_allclose(vecs, np.array([[0.0, 1.0]], dtype=np.float32))
        self.assertEqual(meta, ["b"])
        store.close()

    def test_async_context_manager(self):
        with AsyncFaissVectorStore(dim=2) as store:
            store.add_async(np.array([[1.0, 0.0]])).result()
            vecs, _ = store.search_async(np.array([1.0, 0.0]), k=1).result()
            np.testing.assert_allclose(vecs, np.array([[1.0, 0.0]], dtype=np.float32))

    def test_async_with_context_manager(self):
        async def run():
            async with AsyncFaissVectorStore(dim=2) as store:
                await store.aadd(np.array([[1.0, 0.0]]))
                vecs, _ = await store.asearch(np.array([1.0, 0.0]), k=1)
                np.testing.assert_allclose(vecs, np.array([[1.0, 0.0]], dtype=np.float32))

        asyncio.run(run())

if __name__ == "__main__":
    unittest.main()
