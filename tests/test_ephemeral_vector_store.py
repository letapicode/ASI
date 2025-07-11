import time
import unittest
import numpy as np

from asi.vector_stores import EphemeralVectorStore


class TestEphemeralVectorStore(unittest.TestCase):
    def test_add_and_search(self):
        store = EphemeralVectorStore(dim=2, ttl=1.0)
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]]), metadata=["a", "b"])
        vecs, meta = store.search(np.array([1.0, 0.0]), k=1)
        np.testing.assert_allclose(vecs, np.array([[1.0, 0.0]], dtype=np.float32))
        self.assertEqual(meta, ["a"])

    def test_expiration(self):
        store = EphemeralVectorStore(dim=2, ttl=0.1)
        store.add(np.array([[1.0, 0.0]]), metadata=["x"])
        time.sleep(0.2)
        self.assertEqual(len(store), 0)

    def test_search_after_expire(self):
        store = EphemeralVectorStore(dim=2, ttl=0.1)
        store.add(np.array([[1.0, 0.0]]), metadata=["a"])
        time.sleep(0.05)
        store.add(np.array([[0.0, 1.0]]), metadata=["b"])
        time.sleep(0.07)
        vecs, meta = store.search(np.array([0.0, 1.0]), k=2)
        self.assertEqual(meta, ["b"])
        self.assertEqual(len(vecs), 1)


if __name__ == "__main__":
    unittest.main()
