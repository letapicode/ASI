import time
import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory, push_remote, query_remote
from asi.federated_memory_server import FederatedMemoryServer


class TestFederatedMemoryServer(unittest.TestCase):
    def test_replication(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        mem1 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        mem2 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        s1 = FederatedMemoryServer(mem1, "localhost:50600", peers=["localhost:50601"])
        s2 = FederatedMemoryServer(mem2, "localhost:50601", peers=["localhost:50600"])
        s1.start()
        s2.start()

        vec = torch.randn(1, 4)
        push_remote("localhost:50600", vec[0])
        time.sleep(0.1)
        out, meta = query_remote("localhost:50601", vec[0], k=1)

        s1.stop(0)
        s2.stop(0)
        self.assertEqual(out.shape, (1, 4))
        self.assertEqual(len(meta), 1)


if __name__ == "__main__":
    unittest.main()
