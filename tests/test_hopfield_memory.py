import unittest
import numpy as np

from asi.hopfield_memory import HopfieldMemory


class TestHopfieldMemory(unittest.TestCase):
    def test_store_and_retrieve(self) -> None:
        np.random.seed(0)
        mem = HopfieldMemory(dim=8)
        patterns = np.where(np.random.randn(3, 8) > 0, 1, -1).astype(np.float32)
        for p in patterns:
            mem.store(p)
        noisy = patterns.copy()
        for row in noisy:
            idx = np.random.choice(8, 2, replace=False)
            row[idx] *= -1
        out = mem.retrieve(noisy, steps=10)
        self.assertTrue(np.array_equal(out, patterns))


if __name__ == "__main__":
    unittest.main()
