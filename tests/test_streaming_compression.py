import unittest
import torch

from asi.streaming_compression import (
    ReservoirBuffer,
    StreamingCompressor,
    TemporalVectorCompressor,
)


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


class TestTemporalVectorCompressor(unittest.TestCase):
    def test_decay_replaces_old(self):
        tc = TemporalVectorCompressor(dim=2, compressed_dim=1, capacity=2, decay=0.5)
        first = torch.tensor([[1.0, 0.0]])
        second = torch.tensor([[0.0, 1.0]])
        tc.add(first)
        tc.add(second)
        # add third which should evict first due to decay
        third = torch.tensor([[2.0, 2.0]])
        tc.add(third)
        buf = tc.buffer.get()
        self.assertEqual(buf.shape[0], 2)
        self.assertTrue(any(torch.allclose(b, third[0]) for b in buf))


if __name__ == "__main__":
    unittest.main()
