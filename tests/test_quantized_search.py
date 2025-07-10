import time
import unittest
import numpy as np
import types, sys
sys.modules.setdefault('PIL', types.ModuleType('PIL'))
sys.modules.setdefault('PIL.Image', types.ModuleType('PIL.Image'))
from asi.hierarchical_memory import HierarchicalMemory, torch as ht


class TestQuantizedSearch(unittest.TestCase):
    def test_recall_latency(self) -> None:
        dim = 8
        mem = HierarchicalMemory(dim=dim, compressed_dim=dim, capacity=200, use_pq=True)
        pq = mem.pq_store
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((500, dim)).astype(np.float32)
        metas = list(range(500))
        for v, m in zip(vecs, metas):
            t = ht.from_numpy(v)
            mem.add(t, metadata=[m])
        # pq already populated via add
        q = ht.from_numpy(rng.standard_normal(dim).astype(np.float32))
        start = time.time()
        v1, m1 = mem.search(q, k=1)
        pq_time = time.time() - start

        start = time.time()
        exact_vecs, exact_meta = mem.store.search(q.numpy(), k=1)
        exact_time = time.time() - start
        exact_vec = ht.from_numpy(exact_vecs)
        exact_dec = mem.compressor.decoder(exact_vec)[0]
        recall = 1.0 if m1[0] == exact_meta[0] else 0.0
        self.assertGreaterEqual(recall, 0.999)
        self.assertLessEqual(pq_time, exact_time * 2 + 1e-5)


if __name__ == "__main__":
    unittest.main()
