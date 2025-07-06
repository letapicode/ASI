import unittest
import unittest.mock
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.context_summary_memory import ContextSummaryMemory
from asi.bio_memory_replay import BioMemoryReplayer, run_nightly_replay
from asi.distributed_trainer import DistributedTrainer, MemoryConfig
from asi.distributed_memory import DistributedMemory


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.count += x.size(0)
        return x


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
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        data = torch.randn(3, 4)
        mem.add(data, metadata=["a", "b", "c"])
        model = DummyModel()
        replayer = BioMemoryReplayer(model, mem, batch_size=2)
        replayer.replay()
        self.assertEqual(model.count, 3)

    def test_replayer_ctxsum(self):
        mem = ContextSummaryMemory(
            dim=4, compressed_dim=2, capacity=4, summarizer=DummySummarizer(), context_size=1
        )
        data = torch.randn(2, 4)
        mem.add(data, metadata=["x", "y"])
        mem.summarize_context()
        replayer = BioMemoryReplayer(DummyModel(), mem)
        seqs = replayer.reconstruct_sequences()
        self.assertTrue(torch.allclose(seqs[0], torch.ones(4)))

    def test_run_nightly_replay(self):
        cfg = MemoryConfig(dim=4, compressed_dim=2, capacity=10)
        with unittest.mock.patch("time.time", side_effect=[0.0, 1.0, 2.0]):
            trainer = DistributedTrainer(simple_worker, cfg, "tmp", max_restarts=1)
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
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
