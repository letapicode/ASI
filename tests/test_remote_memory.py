import unittest
import numpy as np

from asi.hierarchical_memory import HierarchicalMemory, torch as ht
try:
    import torch  # noqa: F401
except Exception:
    torch = None
from asi.memory_service import serve
from asi.memory_clients import RemoteMemoryClient


class TestRemoteMemory(unittest.TestCase):
    def test_add_and_search(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")
        if torch is None:
            self.skipTest("torch not available")

        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        server = serve(mem, "localhost:50200")

        client = RemoteMemoryClient("localhost:50200")
        data = ht.from_numpy(np.random.randn(1, 4).astype(np.float32))
        client.add(data[0], metadata=["x"])
        out, meta = client.search(data[0], k=1)
        self.assertEqual(out.shape, (1, 4))
        self.assertEqual(meta[0], "x")
        server.stop(0)

    def test_batch_add_and_search(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")
        if torch is None:
            self.skipTest("torch not available")

        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        server = serve(mem, "localhost:50201")

        client = RemoteMemoryClient("localhost:50201")
        data = ht.from_numpy(np.random.randn(2, 4).astype(np.float32))
        client.add_batch(data, metadata=["a", "b"])
        out, meta = client.query_batch(data, k=1)
        self.assertEqual(out.shape, (2, 1, 4))
        self.assertEqual(len(meta), 2)
        server.stop(0)


if __name__ == "__main__":
    unittest.main()
