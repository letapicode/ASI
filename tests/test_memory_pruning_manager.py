import unittest
import importlib.machinery
import importlib.util
import types
import sys
from pathlib import Path
import torch

pkg = types.ModuleType("asi")
pkg.__path__ = [str(Path("src").resolve())]
sys.modules["asi"] = pkg

# Stub out cryptography dependency required by encrypted_vector_store
crypto = types.ModuleType("cryptography")
hazmat = types.ModuleType("hazmat")
primitives = types.ModuleType("primitives")
ciphers = types.ModuleType("ciphers")
aead = types.ModuleType("aead")

class DummyAESGCM:  # minimal stub
    def __init__(self, *args, **kwargs) -> None:
        pass

aead.AESGCM = DummyAESGCM
ciphers.aead = aead
primitives.ciphers = ciphers
hazmat.primitives = primitives
crypto.hazmat = hazmat
for name, mod in {
    "cryptography": crypto,
    "cryptography.hazmat": hazmat,
    "cryptography.hazmat.primitives": primitives,
    "cryptography.hazmat.primitives.ciphers": ciphers,
    "cryptography.hazmat.primitives.ciphers.aead": aead,
}.items():
    sys.modules.setdefault(name, mod)

psutil_stub = types.ModuleType("psutil")
psutil_stub.cpu_percent = lambda interval=None: 0.0
psutil_stub.virtual_memory = lambda: types.SimpleNamespace(percent=0.0)
psutil_stub.net_io_counters = lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0)
psutil_stub.sensors_battery = lambda: None
sys.modules.setdefault("psutil", psutil_stub)

# Provide minimal stubs for optional ASI modules
stub_mods = {
    "asi.data_ingest": types.ModuleType("data_ingest"),
    "asi.retrieval_rl": types.ModuleType("retrieval_rl"),
    "asi.user_preferences": types.ModuleType("user_preferences"),
    "asi.pq_vector_store": types.ModuleType("pq_vector_store"),
    "asi.async_vector_store": types.ModuleType("async_vector_store"),
}
stub_mods["asi.data_ingest"].CrossLingualTranslator = object
stub_mods["asi.retrieval_rl"].RetrievalPolicy = object
stub_mods["asi.user_preferences"].UserPreferences = object
class _DummyAsyncStore:
    pass

stub_mods["asi.pq_vector_store"].PQVectorStore = object
stub_mods["asi.async_vector_store"].AsyncFaissVectorStore = _DummyAsyncStore
for name, mod in stub_mods.items():
    sys.modules.setdefault(name, mod)


def _load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


HierarchicalMemory = _load("asi.hierarchical_memory", "src/hierarchical_memory.py").HierarchicalMemory
MemoryPruningManager = _load("asi.memory_pruning_manager", "src/memory_pruning_manager.py").MemoryPruningManager
TelemetryLogger = _load("asi.telemetry", "src/telemetry.py").TelemetryLogger


class DummySummarizer:
    def summarize(self, x):
        return "s"


class TestMemoryPruningManager(unittest.TestCase):
    def test_prune_unused(self):
        tel = TelemetryLogger()
        pruner = MemoryPruningManager(threshold=1, telemetry=tel)
        mem = HierarchicalMemory(
            dim=2,
            compressed_dim=1,
            capacity=4,
            pruner=pruner,
            encryption_key=b"0" * 16,
        )
        data = torch.randn(2, 2)
        mem.add(data, metadata=["a", "b"])
        mem.search(data[0], k=1)  # use only 'a'
        pruner.prune()
        self.assertEqual(len(mem), 1)
        self.assertIn("a", mem.store._meta)
        self.assertTrue(any(e.get("event") == "memory_prune" for e in tel.events))

    def test_replace_with_summary(self):
        pruner = MemoryPruningManager(threshold=1, summarizer=DummySummarizer())
        mem = HierarchicalMemory(
            dim=2,
            compressed_dim=1,
            capacity=2,
            pruner=pruner,
            encryption_key=b"0" * 16,
        )
        data = torch.randn(1, 2)
        mem.add(data, metadata=["x"])
        pruner.prune()
        self.assertIsInstance(mem.store._meta[0], dict)
        self.assertEqual(mem.store._meta[0].get("summary"), "s")


if __name__ == "__main__":
    unittest.main()

