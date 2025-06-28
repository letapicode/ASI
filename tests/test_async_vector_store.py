import os
import tempfile
import unittest
import numpy as np

from asi.async_vector_store import AsyncFaissVectorStore


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

if __name__ == "__main__":
    unittest.main()
