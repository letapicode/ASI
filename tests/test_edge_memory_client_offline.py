import time
import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_service import serve
from asi.edge_memory_client import EdgeMemoryClient


class TestEdgeMemoryClientOffline(unittest.TestCase):
    def test_queue_and_sync(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        client = EdgeMemoryClient("localhost:50220", buffer_size=1, sync_interval=0.2)
        data = torch.randn(1, 4)
        client.add(data[0], metadata=["a"])
        time.sleep(0.3)
        server = serve(mem, "localhost:50220")
        time.sleep(0.5)
        out, meta = client.search(data[0], k=1)
        server.stop(0)
        client.close()
        self.assertEqual(out.shape, (1, 4))
        self.assertEqual(meta[0], "a")


if __name__ == "__main__":
    unittest.main()
