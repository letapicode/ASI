import asyncio
import json
import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_profiler import MemoryProfiler


class TestMemoryProfiler(unittest.TestCase):
    def test_stats_collection(self):
        torch.manual_seed(0)
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        profiler = MemoryProfiler(mem)
        profiler.start_profiling()

        # miss before any vectors are added
        mem.search(torch.randn(4), k=1)

        data = torch.randn(1, 4)
        mem.add(data, metadata=["x"])
        mem.search(data[0], k=1)

        stats = json.loads(profiler.report_stats())
        self.assertEqual(stats["queries"], 2)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertGreaterEqual(stats["avg_latency"], 0.0)

    def test_async_stats(self):
        torch.manual_seed(0)

        async def run():
            mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, use_async=True)
            profiler = MemoryProfiler(mem)
            profiler.start_profiling()
            await mem.asearch(torch.randn(4), k=1)  # miss
            data = torch.randn(1, 4)
            await mem.aadd(data, metadata=["y"])
            await mem.asearch(data[0], k=1)  # hit
            csv = profiler.report_stats(fmt="csv")
            parts = csv.strip().split("\n")[1].split(",")
            self.assertEqual(int(parts[0]), 2)
            self.assertEqual(int(parts[1]), 1)
            self.assertEqual(int(parts[2]), 1)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
