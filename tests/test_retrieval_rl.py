import unittest
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.retrieval_rl import RetrievalPolicy, train_policy


class TestRetrievalPolicy(unittest.TestCase):
    def test_update_and_rank(self):
        policy = RetrievalPolicy(epsilon=0.0, alpha=1.0)
        train_policy(policy, [("a", 1.0), ("b", -1.0)])
        order = policy.rank(["b", "a"], [0.5, 0.6])
        self.assertEqual(order[0], 1)

    def test_memory_integration(self):
        policy = RetrievalPolicy(epsilon=0.0, alpha=1.0)
        mem = HierarchicalMemory(dim=2, compressed_dim=2, capacity=10, retrieval_policy=policy)
        good = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
        bad = torch.tensor([[0.8, 0.0]], dtype=torch.float32)
        mem.add(torch.cat([good, bad]), metadata=["good", "bad"])
        q = torch.tensor([1.0, 0.0])
        _, meta = mem.search(q, k=2)
        self.assertEqual(meta[0], "bad")
        train_policy(policy, [("good", 1.0), ("bad", 0.0)])
        out, meta = mem.search(q, k=2)
        self.assertEqual(meta[0], "good")
        self.assertEqual(out.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
