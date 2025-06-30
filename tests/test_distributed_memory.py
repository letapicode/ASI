import unittest
import time
import numpy as np
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.distributed_memory import start_server, push_remote, query_remote


class TestDistributedMemory(unittest.TestCase):
    def test_push_and_query_remote(self):
        torch.manual_seed(0)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        server = start_server(mem, address="localhost:50055")
        try:
            data = np.random.randn(3, 4).astype(np.float32)
            push_remote("localhost:50055", data, metadata=["a", "b", "c"])
            time.sleep(0.1)
            out, meta = query_remote("localhost:50055", data[0], k=1)
            self.assertEqual(out.shape, (1, 4))
            self.assertEqual(len(meta), 1)
            self.assertIn(meta[0], ["a", "b", "c"])
        finally:
            server.stop(0)
            server.wait_for_termination()


if __name__ == "__main__":
    unittest.main()
