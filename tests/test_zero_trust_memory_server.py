import os
import sys
import tempfile
import unittest
import importlib.util
import importlib.machinery
import types

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
        pkg = types.ModuleType("src")
        pkg.__path__ = [SRC_DIR]
        pkg.__spec__ = importlib.machinery.ModuleSpec("src", None, is_package=True)
        sys.modules["src"] = pkg
        spec_pb2 = importlib.util.spec_from_file_location(
            "src.memory_pb2", os.path.join(SRC_DIR, "memory_pb2.py"),
            submodule_search_locations=[SRC_DIR],
        )
        memory_pb2 = importlib.util.module_from_spec(spec_pb2)
        sys.modules["src.memory_pb2"] = memory_pb2
        sys.modules["memory_pb2"] = memory_pb2
        spec_pb2.loader.exec_module(memory_pb2)

        spec_pb2_grpc = importlib.util.spec_from_file_location(
            "src.memory_pb2_grpc", os.path.join(SRC_DIR, "memory_pb2_grpc.py"),
            submodule_search_locations=[SRC_DIR],
        )
        memory_pb2_grpc = importlib.util.module_from_spec(spec_pb2_grpc)
        sys.modules["src.memory_pb2_grpc"] = memory_pb2_grpc
        spec_pb2_grpc.loader.exec_module(memory_pb2_grpc)

        spec_ledger = importlib.util.spec_from_file_location(
            "src.blockchain_provenance_ledger",
            os.path.join(SRC_DIR, "blockchain_provenance_ledger.py"),
            submodule_search_locations=[SRC_DIR],
        )
        ledger_mod = importlib.util.module_from_spec(spec_ledger)
        sys.modules["src.blockchain_provenance_ledger"] = ledger_mod
        spec_ledger.loader.exec_module(ledger_mod)
        BlockchainProvenanceLedger = ledger_mod.BlockchainProvenanceLedger

        spec_zts = importlib.util.spec_from_file_location(
            "src.zero_trust_memory_server",
            os.path.join(SRC_DIR, "zero_trust_memory_server.py"),
            submodule_search_locations=[SRC_DIR],
        )
        zts_mod = importlib.util.module_from_spec(spec_zts)
        sys.modules["src.zero_trust_memory_server"] = zts_mod
        spec_zts.loader.exec_module(zts_mod)
        ZeroTrustMemoryServer = zts_mod.ZeroTrustMemoryServer

        class DummyMemory:
            def __init__(self, dim: int) -> None:
                self.dim = dim
                self.vectors: list[torch.Tensor] = []

            def add(self, vec: torch.Tensor, metadata=None) -> None:
                self.vectors.append(vec.clone())

            def search(self, q: torch.Tensor, k: int = 1):
                if not self.vectors:
                    return torch.zeros(k, self.dim), [None] * k
                out = self.vectors[0].unsqueeze(0).expand(k, -1)
                return out, [None] * k

        mem = DummyMemory(dim=4)
        with tempfile.TemporaryDirectory() as root:
            ledger = BlockchainProvenanceLedger(root)
            ledger.append("valid-token", signature="sig")

            server = ZeroTrustMemoryServer(mem, ledger, address="localhost:50800")
            server.start()

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
