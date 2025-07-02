import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_service import serve
from asi.federated_memory_exchange import FederatedMemoryExchange


class TestFederatedMemoryExchange(unittest.TestCase):
    def test_replication(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        mem1 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        mem2 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        s1 = serve(mem1, "localhost:50500")
        s2 = serve(mem2, "localhost:50501")

        ex1 = FederatedMemoryExchange(mem1, peers=["localhost:50501"])
        ex2 = FederatedMemoryExchange(mem2, peers=["localhost:50500"])

        vec = torch.randn(1, 4)
        ex1.push(vec, metadata=["a"])

        out, meta = ex2.query(vec[0], k=1)
        self.assertEqual(meta[0], "a")

        s1.stop(0)
        s2.stop(0)


if __name__ == "__main__":
    unittest.main()
