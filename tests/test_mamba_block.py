import unittest
import torch

from src.mamba_block import MambaBlock


class TestMambaBlock(unittest.TestCase):
    def test_forward_shape(self):
        block = MambaBlock(dim=4)
        x = torch.randn(2, 5, 4)
        out = block(x)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
