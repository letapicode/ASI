import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np
import tempfile

from src.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    def test_add_and_search(self):
        store = VectorStore(dim=2)
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]]), metadata=["a", "b"])
        vecs, meta = store.search(np.array([1.0, 0.0]), k=1)
        np.testing.assert_allclose(vecs, np.array([[1.0, 0.0]], dtype=np.float32))
        self.assertEqual(meta, ["a"])

    def test_empty(self):
        store = VectorStore(dim=3)
        vecs, meta = store.search(np.array([0.0, 0.0, 1.0]), k=2)
        self.assertEqual(vecs.shape[0], 0)
        self.assertEqual(meta, [])

    def test_dimension_error(self):
        store = VectorStore(dim=3)
        with self.assertRaises(ValueError):
            store.add(np.array([1.0, 2.0]))

    def test_save_load(self):
        store = VectorStore(dim=2)
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]]), metadata=["a", "b"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "store.npz")
            store.save(path)
            loaded = VectorStore.load(path)
            self.assertEqual(len(loaded), 2)
            vecs, meta = loaded.search(np.array([0.0, 1.0]), k=1)
            np.testing.assert_allclose(vecs, np.array([[0.0, 1.0]], dtype=np.float32))
            self.assertEqual(meta, ["b"])

if __name__ == "__main__":
    unittest.main()
