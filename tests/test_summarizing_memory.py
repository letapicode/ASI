import unittest
import torch

from asi.summarizing_memory import SummarizingMemory


class DummySummarizer:
    def __call__(self, x):
        return "sum"


class TestSummarizingMemory(unittest.TestCase):
    def test_summarize(self):
        mem = SummarizingMemory(dim=2, compressed_dim=1, capacity=3)
        data = torch.randn(3, 2)
        mem.add(data, metadata=["a", "b", "c"])
        mem.summarize(DummySummarizer())
        self.assertEqual(len(mem), 3)


if __name__ == "__main__":
    unittest.main()
