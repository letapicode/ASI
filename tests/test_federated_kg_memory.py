import importlib.machinery
import importlib.util
import types
import sys
import time
import unittest

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

KnowledgeGraphMemory = _load('asi.knowledge_graph_memory', 'src/knowledge_graph_memory.py').KnowledgeGraphMemory
_load('asi.kg_memory_pb2', 'src/kg_memory_pb2.py')
sys.modules['kg_memory_pb2'] = sys.modules['asi.kg_memory_pb2']
_load('asi.kg_memory_pb2_grpc', 'src/kg_memory_pb2_grpc.py')
sys.modules['kg_memory_pb2_grpc'] = sys.modules['asi.kg_memory_pb2_grpc']
fkg_mod = _load('asi.federated_kg_memory', 'src/federated_kg_memory.py')
FederatedKGMemoryServer = fkg_mod.FederatedKGMemoryServer
push_triple_remote = fkg_mod.push_triple_remote
query_triples_remote = fkg_mod.query_triples_remote


class TestFederatedKGMemory(unittest.TestCase):
    def test_replication(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        kg1 = KnowledgeGraphMemory()
        kg2 = KnowledgeGraphMemory()
        s1 = FederatedKGMemoryServer(kg1, "localhost:50800", peers=["localhost:50801"])
        s2 = FederatedKGMemoryServer(kg2, "localhost:50801", peers=["localhost:50800"])
        s1.start()
        s2.start()

        push_triple_remote("localhost:50800", ("x", "y", "z"))
        time.sleep(0.1)
        res = query_triples_remote("localhost:50801", subject="x")

        s1.stop(0)
        s2.stop(0)
        self.assertIn(("x", "y", "z"), res)

    def test_last_write_wins(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        kg1 = KnowledgeGraphMemory()
        kg2 = KnowledgeGraphMemory()
        s1 = FederatedKGMemoryServer(kg1, "localhost:50802", peers=["localhost:50803"])
        s2 = FederatedKGMemoryServer(kg2, "localhost:50803", peers=["localhost:50802"])
        s1.start()
        s2.start()

        push_triple_remote("localhost:50802", ("a", "b", "c1"), key="k")
        push_triple_remote("localhost:50803", ("a", "b", "c2"), key="k")
        time.sleep(0.2)
        r1 = query_triples_remote("localhost:50802", subject="a")
        r2 = query_triples_remote("localhost:50803", subject="a")

        s1.stop(0)
        s2.stop(0)
        self.assertEqual(r1, r2)
        self.assertEqual(r1[0][2], "c2")


if __name__ == "__main__":
    unittest.main()
