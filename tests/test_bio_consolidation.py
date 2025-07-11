import unittest
import unittest.mock
import torch
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType("asi")
pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules["asi"] = pkg

def _load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

_load("asi.streaming_compression", "src/streaming_compression.py")
_load("asi.vector_stores", "src/vector_stores.py")
HierarchicalMemory = _load("asi.hierarchical_memory", "src/hierarchical_memory.py").HierarchicalMemory
BioMemoryReplayer = _load("asi.bio_memory_replay", "src/bio_memory_replay.py").BioMemoryReplayer
dt_mod = _load("asi.distributed_trainer", "src/distributed_trainer.py")
DistributedTrainer = dt_mod.DistributedTrainer
MemoryConfig = dt_mod.MemoryConfig
DistributedMemory = _load("asi.distributed_memory", "src/distributed_memory.py").DistributedMemory


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.count += x.size(0)
        return self.lin(x)


def simple_worker(memory: DistributedMemory, step: int, compress=None) -> None:
    tensor = torch.ones(1, memory.compressor.encoder.in_features)
    if compress is not None:
        tensor = compress(tensor)
    memory.add(tensor, metadata=[f"s{step}"])


class TestBioConsolidation(unittest.TestCase):
    def test_reinsertion_compressed(self):
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, encryption_key=b"0"*16)
        data = torch.randn(2, 4)
        mem.add(data, metadata=["a", "b"])
        before_len = len(mem)
        before_buf = len(mem.compressor.buffer.data)
        model = DummyModel()
        replayer = BioMemoryReplayer(model, mem, batch_size=1)
        replayer.replay()
        self.assertEqual(len(mem), before_len + 2)
        self.assertEqual(len(mem.compressor.buffer.data), before_buf)

    def test_consolidation_hook(self):
        cfg = MemoryConfig(dim=4, compressed_dim=2, capacity=10)
        trainer = DistributedTrainer(simple_worker, cfg, "tmp", max_restarts=1)
        called = []

        def hook():
            called.append(True)

        trainer.consolidation_hook = hook
        trainer.consolidation_interval = 0.0
        with unittest.mock.patch("multiprocessing.Process") as proc:
            proc.return_value.start = lambda: None
            proc.return_value.join = lambda: None
            proc.return_value.exitcode = 0
            trainer.run(steps=1)
        self.assertTrue(called)


if __name__ == "__main__":
    unittest.main()
