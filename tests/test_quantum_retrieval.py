import unittest
import time
import numpy as np

from asi.vector_stores import FaissVectorStore


class TestQuantumRetrieval(unittest.TestCase):
    def test_benchmark(self):
        rng = np.random.default_rng(0)
        dim = 16
        vecs = rng.normal(size=(200, dim)).astype(np.float32)
        store = FaissVectorStore(dim=dim)
        store.add(vecs, metadata=list(range(len(vecs))))

        queries = vecs[:20] + rng.normal(scale=0.01, size=(20, dim)).astype(np.float32)

        start = time.perf_counter()
        classical_hits = 0
        for i, q in enumerate(queries):
            _, meta = store.search(q, k=1)
            if meta and meta[0] == i:
                classical_hits += 1
        classical_time = time.perf_counter() - start

        start = time.perf_counter()
        quantum_hits = 0
        for i, q in enumerate(queries):
            _, meta = store.search(q, k=1, quantum=True)
            if meta and meta[0] == i:
                quantum_hits += 1
        quantum_time = time.perf_counter() - start

        self.assertGreaterEqual(quantum_hits, classical_hits - 2)
        self.assertLessEqual(quantum_time, classical_time * 2)


if __name__ == "__main__":
    unittest.main()
