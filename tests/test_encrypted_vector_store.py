import os
import sys
import importlib.util
import tempfile
import unittest
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
spec_vs = importlib.util.spec_from_file_location(
    "src.vector_store", os.path.join(SRC_DIR, "vector_store.py")
)
vector_store = importlib.util.module_from_spec(spec_vs)
sys.modules["src.vector_store"] = vector_store
spec_vs.loader.exec_module(vector_store)

spec_enc = importlib.util.spec_from_file_location(
    "src.encrypted_vector_store",
    os.path.join(SRC_DIR, "encrypted_vector_store.py"),
    submodule_search_locations=[SRC_DIR],
)
encrypted_vector_store = importlib.util.module_from_spec(spec_enc)
sys.modules["src.encrypted_vector_store"] = encrypted_vector_store
spec_enc.loader.exec_module(encrypted_vector_store)

EncryptedVectorStore = encrypted_vector_store.EncryptedVectorStore
VectorStore = vector_store.VectorStore


class TestEncryptedVectorStore(unittest.TestCase):
    def test_save_load_encrypted(self):
        key = b"0" * 32
        store = EncryptedVectorStore(dim=2, key=key)
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]]), metadata=["a", "b"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "store.enc")
            store.save(path)
            loaded = EncryptedVectorStore.load(path, key)
            q = np.array([0.0, 1.0], dtype=np.float32)
            vecs_e, meta_e = loaded.search(q, k=1)
        base = VectorStore(dim=2)
        base.add(np.array([[1.0, 0.0], [0.0, 1.0]]), metadata=["a", "b"])
        vecs_b, meta_b = base.search(q, k=1)
        np.testing.assert_allclose(vecs_e, vecs_b)
        self.assertEqual(meta_e, meta_b)

    def test_hierarchical_memory_integration(self):
        if torch is None:
            self.skipTest("torch not available")
        torch.manual_seed(0)
        spec_hm = importlib.util.spec_from_file_location(
            "src.hierarchical_memory",
            os.path.join(SRC_DIR, "hierarchical_memory.py"),
            submodule_search_locations=[SRC_DIR],
        )
        hierarchical_memory = importlib.util.module_from_spec(spec_hm)
        sys.modules["src.hierarchical_memory"] = hierarchical_memory
        spec_hm.loader.exec_module(hierarchical_memory)
        HierarchicalMemory = hierarchical_memory.HierarchicalMemory

        key = b"1" * 32
        mem = HierarchicalMemory(
            dim=4, compressed_dim=2, capacity=10, encryption_key=key
        )
        data = torch.randn(3, 4)
        mem.add(data, metadata=["x", "y", "z"])
        q = data[0]
        out, meta = mem.search(q, k=1)
        self.assertEqual(out.shape, (1, 4))
        self.assertIn(meta[0], ["x", "y", "z"])


if __name__ == "__main__":
    unittest.main()
