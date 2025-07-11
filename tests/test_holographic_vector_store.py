import unittest
import numpy as np
from asi.vector_stores import HolographicVectorStore


class TestHolographicVectorStore(unittest.TestCase):
    def test_encode_decode(self):
        store = HolographicVectorStore(dim=8)
        t = np.random.randn(8).astype(np.float32)
        i = np.random.randn(8).astype(np.float32)
        a = np.random.randn(8).astype(np.float32)
        enc = store.encode(t, i, a)
        dt, di, da = store.decode(enc)
        np.testing.assert_allclose(dt, t, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(di, i, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(da, a, rtol=1e-5, atol=1e-5)

    def test_add_search(self):
        store = HolographicVectorStore(dim=8)
        t = np.random.randn(8).astype(np.float32)
        i = np.random.randn(8).astype(np.float32)
        a = np.random.randn(8).astype(np.float32)
        vec = store.encode(t, i, a)
        store.add(vec[None], metadata=["x"])
        q = store.encode(t, i, a)
        vecs, meta = store.search(q, k=1)
        self.assertEqual(meta, ["x"])
        self.assertEqual(vecs.shape, (1, 8))

    def test_encode_batch(self):
        store = HolographicVectorStore(dim=8)
        t = np.random.randn(3, 8).astype(np.float32)
        i = np.random.randn(3, 8).astype(np.float32)
        a = np.random.randn(3, 8).astype(np.float32)
        enc = store.encode_batch(t, i, a)
        self.assertEqual(enc.shape, (3, 8))
        dt, di, da = store.decode(enc[0])
        np.testing.assert_allclose(dt, t[0], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
