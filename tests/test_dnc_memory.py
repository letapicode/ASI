import os
import unittest
import torch

from asi.dnc_memory import DNCMemory


class TestDNCMemory(unittest.TestCase):
    def test_store_and_retrieve(self):
        torch.manual_seed(0)
        mem = DNCMemory(memory_size=4, word_size=3)
        data = torch.randn(2, 3)
        mem.write(data, metadata=["a", "b"])
        vec, meta = mem.read(data[0], k=1)
        self.assertEqual(meta[0], "a")
        self.assertEqual(vec.shape, (1, 3))
        vec2, meta2 = mem.read(data[1], k=1)
        self.assertEqual(meta2[0], "b")
        self.assertEqual(vec2.shape, (1, 3))

    def test_save_and_load(self):
        torch.manual_seed(1)
        mem = DNCMemory(memory_size=2, word_size=3)
        data = torch.randn(1, 3)
        mem.add(data, metadata=["x"])
        path = "tmp_dnc.npz"
        mem.save(path)
        mem2 = DNCMemory.load(path)
        vec, meta = mem2.search(data[0], k=1)
        self.assertEqual(meta[0], "x")
        self.assertTrue(torch.allclose(vec, data, atol=1e-6))
        os.remove(path)

    def test_reset(self):
        mem = DNCMemory(memory_size=2, word_size=2)
        mem.add(torch.ones(2, 2), metadata=["a", "b"])
        self.assertEqual(len(mem), 2)
        mem.reset()
        self.assertEqual(len(mem), 0)


if __name__ == "__main__":
    unittest.main()
