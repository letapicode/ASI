import os
import tempfile
import unittest
import importlib.util

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional
    torch = None
    _HAS_TORCH = False

try:
    import grpc  # noqa: F401
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional
    grpc = None
    _HAS_GRPC = False

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")



class TestZeroTrustMemoryServer(unittest.TestCase):
    def test_unauthorized_rejected(self):
        if not (_HAS_GRPC and _HAS_TORCH):
            self.skipTest("dependencies not available")
        spec_hm = importlib.util.spec_from_file_location(
            "src.hierarchical_memory", os.path.join(SRC_DIR, "hierarchical_memory.py"),
            submodule_search_locations=[SRC_DIR],
        )
        hierarchical_memory = importlib.util.module_from_spec(spec_hm)
        spec_hm.loader.exec_module(hierarchical_memory)
        HierarchicalMemory = hierarchical_memory.HierarchicalMemory

        spec_ms = importlib.util.spec_from_file_location(
            "src.memory_service", os.path.join(SRC_DIR, "memory_service.py"),
            submodule_search_locations=[SRC_DIR],
        )
        memory_service = importlib.util.module_from_spec(spec_ms)
        spec_ms.loader.exec_module(memory_service)
        serve = memory_service.serve

        spec_ledger = importlib.util.spec_from_file_location(
            "src.blockchain_provenance_ledger",
            os.path.join(SRC_DIR, "blockchain_provenance_ledger.py"),
            submodule_search_locations=[SRC_DIR],
        )
        ledger_mod = importlib.util.module_from_spec(spec_ledger)
        spec_ledger.loader.exec_module(ledger_mod)
        BlockchainProvenanceLedger = ledger_mod.BlockchainProvenanceLedger

        spec_pb2 = importlib.util.spec_from_file_location(
            "src.memory_pb2", os.path.join(SRC_DIR, "memory_pb2.py"),
            submodule_search_locations=[SRC_DIR],
        )
        memory_pb2 = importlib.util.module_from_spec(spec_pb2)
        spec_pb2.loader.exec_module(memory_pb2)

        spec_pb2_grpc = importlib.util.spec_from_file_location(
            "src.memory_pb2_grpc", os.path.join(SRC_DIR, "memory_pb2_grpc.py"),
            submodule_search_locations=[SRC_DIR],
        )
        memory_pb2_grpc = importlib.util.module_from_spec(spec_pb2_grpc)
        spec_pb2_grpc.loader.exec_module(memory_pb2_grpc)

        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        with tempfile.TemporaryDirectory() as root:
            ledger = BlockchainProvenanceLedger(root)
            ledger.append("valid-token", signature="sig")

            server = serve(mem, "localhost:50800", ledger=ledger)

            vec = torch.randn(1, 4)
            req = memory_pb2.PushRequest(vector=vec[0].tolist(), metadata="")

            with grpc.insecure_channel("localhost:50800") as channel:
                stub = memory_pb2_grpc.MemoryServiceStub(channel)
                with self.assertRaises(grpc.RpcError):
                    stub.Push(req, timeout=5, metadata=(("authorization", "bad"),))
                reply = stub.Push(
                    req, timeout=5, metadata=(("authorization", "valid-token"),)
                )
                self.assertTrue(reply.ok)

            server.stop(0)


if __name__ == "__main__":
    unittest.main()
