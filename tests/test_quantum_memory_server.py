import unittest
import numpy as np

from asi.vector_store import VectorStore
from asi.quantum_memory_server import QuantumMemoryServer
from asi.memory_clients import QuantumMemoryClient


class TestQuantumMemoryServer(unittest.TestCase):
    def test_add_and_query(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        store = VectorStore(dim=4)
        server = QuantumMemoryServer(store, "localhost:50800")
        server.start()

        client = QuantumMemoryClient("localhost:50800")
        vec = np.random.randn(4).astype(np.float32)
        client.add(vec, metadata="a")
        out, meta = client.search(vec, k=1)

        server.stop(0)
        client.close()

        np.testing.assert_allclose(out, vec.reshape(1, -1), atol=1e-5)
        self.assertEqual(meta, ["a"])


if __name__ == "__main__":
    unittest.main()
