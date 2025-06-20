import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from src.streaming_compression import ReservoirBuffer, StreamingCompressor


class TestReservoirBuffer(unittest.TestCase):
    def test_capacity(self):
        buf = ReservoirBuffer(capacity=3)
        data = torch.arange(10).float().view(10, 1)
        buf.add(data)
        stored = buf.get()
        self.assertEqual(stored.shape[0], 3)


class TestStreamingCompressor(unittest.TestCase):
    def test_compress_reconstruct(self):
        sc = StreamingCompressor(dim=4, compressed_dim=2, capacity=5)
        x = torch.randn(8, 4)
        sc.add(x)
        comp = sc.compressed()
        self.assertEqual(comp.shape, (5, 2))
        recon = sc.reconstruct()
        self.assertEqual(recon.shape, (5, 4))


if __name__ == "__main__":
    unittest.main()
