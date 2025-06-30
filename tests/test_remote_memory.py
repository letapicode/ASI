import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_service import serve
from asi.remote_memory import RemoteMemory


class TestRemoteMemory(unittest.TestCase):
    def test_add_and_search(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        server = serve(mem, "localhost:50200")

        client = RemoteMemory("localhost:50200")
        data = torch.randn(1, 4)
        client.add(data[0], metadata=["x"])
        out, meta = client.search(data[0], k=1)
        self.assertEqual(out.shape, (1, 4))
        self.assertEqual(meta[0], "x")
        server.stop(0)


if __name__ == "__main__":
    unittest.main()
