import os
import tempfile
import unittest
from pathlib import Path

import torch

from asi.distributed_trainer import (
    DistributedTrainer,
    MemoryConfig,
    GradientCompressionConfig,
    EnclaveConfig,
)
from asi.distributed_memory import DistributedMemory
from unittest.mock import patch


def flaky_train(memory: DistributedMemory, step: int, compress=None) -> None:
    """Simulate a worker that fails once then succeeds."""
    marker = os.environ.get("FAIL_ONCE")
    if marker and not Path(marker).exists():
        Path(marker).write_text("x")
        raise RuntimeError("fail once")
    tensor = torch.ones(1, memory.compressor.encoder.in_features) * (step + 1)
    if compress is not None:
        tensor = compress(tensor)
    memory.add(tensor, metadata=[f"s{step}"])


class TestDistributedTrainer(unittest.TestCase):
    def test_restart_and_checkpoint(self):
        cfg = MemoryConfig(dim=4, compressed_dim=2, capacity=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["FAIL_ONCE"] = str(Path(tmpdir) / "marker")
            trainer = DistributedTrainer(flaky_train, cfg, tmpdir, max_restarts=2)
            trainer.run(steps=2)
            ckpt1 = Path(tmpdir) / "step1" / "memory" / "compressor.pt"
            ckpt2 = Path(tmpdir) / "step2" / "memory" / "compressor.pt"
            self.assertTrue(ckpt1.exists())
            self.assertTrue(ckpt2.exists())
        mem = DistributedMemory.load(Path(tmpdir) / "step2" / "memory")
        self.assertGreaterEqual(len(mem), 1)

    def test_gradient_compression(self):
        cfg = MemoryConfig(dim=4, compressed_dim=2, capacity=10)
        comp = GradientCompressionConfig(topk=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DistributedTrainer(
                flaky_train,
                cfg,
                tmpdir,
                max_restarts=1,
                grad_compression=comp,
            )
            trainer.run(steps=1)
            mem = DistributedMemory.load(Path(tmpdir) / "step1" / "memory")
            vec = mem.compressor.buffer.data[0]
            self.assertEqual(int((vec != 0).sum()), 1)


    def test_hpc_dispatch(self):
        cfg = MemoryConfig(dim=4, compressed_dim=2, capacity=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('asi.distributed_trainer.submit_job') as submit:
                trainer = DistributedTrainer(flaky_train, cfg, tmpdir, hpc_backend='slurm')
                trainer.run(steps=1)
                submit.assert_called()

    def test_enclave_runner(self):
        cfg = MemoryConfig(dim=4, compressed_dim=2, capacity=10)
        encl = EnclaveConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DistributedTrainer(
                flaky_train,
                cfg,
                tmpdir,
                max_restarts=1,
                enclave=encl,
            )
            trainer.run(steps=1)
            mem = DistributedMemory.load(Path(tmpdir) / "step1" / "memory")
            self.assertGreaterEqual(len(mem), 1)

    def test_async_mode(self):
        cfg = MemoryConfig(dim=4, compressed_dim=2, capacity=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DistributedTrainer(
                flaky_train,
                cfg,
                tmpdir,
                max_restarts=1,
                async_mode=True,
                async_workers=2,
                sync_steps=2,
            )
            trainer.run(steps=2)
            mem = DistributedMemory.load(Path(tmpdir) / "step2" / "memory")
            self.assertGreaterEqual(mem.compressor.buffer.count, 4)



if __name__ == "__main__":
    unittest.main()
