import time
import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.remote_memory import push_remote, query_remote
from asi.federated_memory_server import FederatedMemoryServer
from asi import memory_pb2, memory_pb2_grpc
import grpc


class TestFederatedMemoryServerProof(unittest.TestCase):
    def test_sync_with_proof(self):
        mem1 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        mem2 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        s1 = FederatedMemoryServer(
            mem1, "localhost:50700", peers=["localhost:50701"], require_proof=True
        )
        s2 = FederatedMemoryServer(
            mem2, "localhost:50701", peers=["localhost:50700"], require_proof=True
        )
        s1.start()
        s2.start()

        vec = torch.randn(1, 4)
        push_remote("localhost:50700", vec[0])
        time.sleep(0.1)
        out, _ = query_remote("localhost:50701", vec[0], k=1)

        s1.stop(0)
        s2.stop(0)
        self.assertEqual(out.shape, (1, 4))

    def test_bad_proof(self):
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        srv = FederatedMemoryServer(mem, "localhost:50702", require_proof=True)
        srv.start()

        with grpc.insecure_channel("localhost:50702") as ch:
            stub = memory_pb2_grpc.MemoryServiceStub(ch)
            entry = memory_pb2.VectorEntry(
                id="a", vector=[0, 0, 0, 0], metadata="a", timestamp=0, proof="bad"
            )
            with self.assertRaises(grpc.RpcError):
                stub.Sync(memory_pb2.SyncRequest(items=[entry]))

        srv.stop(0)


if __name__ == "__main__":
    unittest.main()
