import unittest
import torch

from asi.context_summary_memory import ContextSummaryMemory


class DummySummarizer:
    def summarize(self, x):
        return "s"

    def expand(self, text):
        return torch.ones(2)


class TestContextSummaryMemory(unittest.TestCase):
    def test_summarize_and_expand(self):
        mem = ContextSummaryMemory(
            dim=2, compressed_dim=1, capacity=4, summarizer=DummySummarizer(), context_size=1
        )
        data = torch.randn(3, 2)
        mem.add(data, metadata=["a", "b", "c"])
        mem.summarize_context()
        vecs, meta = mem.search(data[0], k=2)
        self.assertEqual(vecs.shape[0], len(meta))
        self.assertTrue(any(isinstance(m, str) and m.startswith("ctxsum:") for m in meta))
        self.assertTrue(torch.allclose(vecs[0], torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
