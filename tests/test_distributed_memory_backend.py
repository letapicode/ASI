import time
import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.distributed_memory_backend import (
    DistributedMemoryServer,
    push_remote,
    query_remote,
)


class TestDistributedMemoryBackend(unittest.TestCase):
    def test_multi_node_sharing(self):
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        server = DistributedMemoryServer(mem, "localhost:50051")
        server.start()
        # give server time to bind
        time.sleep(0.1)
        try:
            v1 = torch.randn(4)
            v2 = torch.randn(4)
            push_remote("localhost:50051", v1, meta="n1")
            push_remote("localhost:50051", v2, meta="n2")
            out, meta = query_remote("localhost:50051", v1, k=2)
            self.assertEqual(out.shape, (2, 4))
            self.assertEqual(len(meta), 2)
            self.assertIn("n1", meta)
            self.assertIn("n2", meta)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()

