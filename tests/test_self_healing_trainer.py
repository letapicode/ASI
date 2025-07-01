import unittest
import multiprocessing as mp
import torch

from asi.self_healing_trainer import SelfHealingTrainer
from asi.distributed_memory import DistributedMemory
from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_service import serve


class TestSelfHealingTrainer(unittest.TestCase):
    def test_restart_failed_worker(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        remote = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        server = serve(remote, "localhost:50110")
        manager = mp.Manager()
        flags = manager.dict(fail=True)

        def worker(rank: int, world_size: int, addr: str, flags_dict):
            mem = DistributedMemory(dim=4, compressed_dim=2, capacity=10, remotes=[addr])
            if rank == 0 and flags_dict.get("fail", False):
                flags_dict["fail"] = False
                raise RuntimeError("boom")
            x = torch.randn(1, 4)
            mem.add(x, metadata=[f"r{rank}"])

        trainer = SelfHealingTrainer(
            worker_fn=worker,
            world_size=2,
            max_restarts=1,
            addr="localhost:50110",
            flags_dict=flags,
        )
        trainer.run()
        server.stop(0)
        self.assertEqual(len(remote), 2)


if __name__ == "__main__":
    unittest.main()
