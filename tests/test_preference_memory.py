import unittest
import numpy as np
import torch
from asi.hierarchical_memory import HierarchicalMemory
from asi.user_preferences import UserPreferences

class TestPreferenceMemory(unittest.TestCase):
    def test_preference_ranking(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=2, capacity=10)
        vecs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        mem.add(vecs, metadata=["a", "b"])

        prefs = UserPreferences(dim=2)
        prefs.update("u", np.array([0.0, 1.0], dtype=np.float32))

        q = torch.tensor([0.8, 0.6], dtype=torch.float32)
        _, meta_default = mem.search(q, k=2)
        self.assertEqual(meta_default[0], "a")

        _, meta_pref = mem.search(q, k=2, preferences=prefs)
        self.assertEqual(meta_pref[0], "b")

if __name__ == "__main__":
    unittest.main()
