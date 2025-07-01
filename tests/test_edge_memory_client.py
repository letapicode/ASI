import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_service import serve
from asi.edge_memory_client import EdgeMemoryClient


class TestEdgeMemoryClient(unittest.TestCase):
    def test_stream_add_and_query(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        server = serve(mem, "localhost:50210")

        client = EdgeMemoryClient("localhost:50210", buffer_size=2)
        data = torch.randn(3, 4)
        metas = ["a", "b", "c"]

        for vec, m in zip(data, metas):
            client.add(vec, metadata=[m])

        out, meta = client.search(data[0], k=1)
        self.assertEqual(out.shape, (1, 4))
        self.assertIn(meta[0], metas)

        client.close()
        server.stop(0)


if __name__ == "__main__":
    unittest.main()
