import os
import tempfile
import unittest
import numpy as np

from asi.vector_store import VectorStore, FaissVectorStore

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

    def test_save_and_load(self):
        store = VectorStore(dim=2)
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]]), metadata=["a", "b"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "store.npz")
            store.save(path)
            loaded = VectorStore.load(path)
            vecs, meta = loaded.search(np.array([0.0, 1.0]), k=1)
            np.testing.assert_allclose(vecs, np.array([[0.0, 1.0]], dtype=np.float32))
            self.assertEqual(meta, ["b"])

    def test_faiss_persistence(self):
        store = FaissVectorStore(dim=2)
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]]), metadata=["a", "b"])
        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            loaded = FaissVectorStore.load(tmpdir)
            vecs, meta = loaded.search(np.array([0.0, 1.0]), k=1)
            np.testing.assert_allclose(vecs, np.array([[0.0, 1.0]], dtype=np.float32))
            self.assertEqual(meta, ["b"])

if __name__ == "__main__":
    unittest.main()
