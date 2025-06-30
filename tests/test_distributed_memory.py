import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory, MemoryServer
from asi.distributed_memory import DistributedMemory
from asi.memory_service import serve


class TestDistributedMemory(unittest.TestCase):
    def test_replication_and_query(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        remote = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        server = serve(remote, "localhost:50100")

        mem = DistributedMemory(dim=4, compressed_dim=2, capacity=10, remotes=["localhost:50100"])
        data = torch.randn(1, 4)
        mem.add(data, metadata=["x"])

        out_local, meta_local = mem.search(data[0], k=1)
        self.assertEqual(meta_local[0], "x")

        out_remote, meta_remote = remote.search(data[0], k=1)
        self.assertEqual(meta_remote[0], "x")

        server.stop(0)


if __name__ == "__main__":
    unittest.main()
