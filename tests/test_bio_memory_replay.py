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
ContextSummaryMemory = _load("asi.context_summary_memory", "src/context_summary_memory.py").ContextSummaryMemory
bm = _load("asi.bio_memory_replay", "src/bio_memory_replay.py")
BioMemoryReplayer = bm.BioMemoryReplayer
run_nightly_replay = bm.run_nightly_replay
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


class DummySummarizer:
    def summarize(self, x):
        return "s"

    def expand(self, text):
        return torch.ones(4)


def simple_worker(memory: DistributedMemory, step: int, compress=None) -> None:
    tensor = torch.ones(1, memory.compressor.encoder.in_features) * (step + 1)
    if compress is not None:
        tensor = compress(tensor)
    memory.add(tensor, metadata=[f"s{step}"])


class TestBioMemoryReplay(unittest.TestCase):
    def test_replayer_hierarchical(self):
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, encryption_key=b"0"*16)
        data = torch.randn(3, 4)
        mem.add(data, metadata=["a", "b", "c"])
        model = DummyModel()
        replayer = BioMemoryReplayer(model, mem, batch_size=2)
        replayer.replay()
        self.assertEqual(model.count, 3)


    def test_run_nightly_replay(self):
        cfg = MemoryConfig(dim=4, compressed_dim=2, capacity=10)
        with unittest.mock.patch("time.time", side_effect=[0.0, 1.0, 2.0, 3.0]):
            trainer = DistributedTrainer(simple_worker, cfg, "tmp", max_restarts=1)
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, encryption_key=b"0"*16)
            mem.add(torch.randn(1, 4), metadata=["z"])
            model = DummyModel()
            run_nightly_replay(trainer, model, mem, batch_size=1)
            trainer.replay_interval = 0.0
            with unittest.mock.patch("multiprocessing.Process") as proc:
                proc.return_value.start = lambda: None
                proc.return_value.join = lambda: None
                proc.return_value.exitcode = 0
                trainer.run(steps=1)
            self.assertEqual(model.count, 1)


if __name__ == "__main__":
    unittest.main()
