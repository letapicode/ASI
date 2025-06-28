import unittest
import torch

from asi.mamba_block import MambaBlock


class TestMambaBlock(unittest.TestCase):
    def test_forward_shape(self):
        block = MambaBlock(dim=4)
        x = torch.randn(2, 5, 4)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_dropout(self):
        torch.manual_seed(0)
        block = MambaBlock(dim=4, dropout=0.5)
        x = torch.randn(2, 3, 4)
        block.eval()
        out1 = block(x)
        out2 = block(x)
        self.assertTrue(torch.allclose(out1, out2))
        block.train()
        out3 = block(x)
        out4 = block(x)
        self.assertFalse(torch.allclose(out3, out4))

    def test_gating_extremes(self):
        block = MambaBlock(dim=2)
        x = torch.randn(1, 2, 2)
        with torch.no_grad():
            block.gate.weight.fill_(-100)
            block.gate.bias.fill_(-100)
        out_input = block(x)
        with torch.no_grad():
            block.gate.weight.fill_(100)
            block.gate.bias.fill_(100)
        out_state = block(x)
        self.assertFalse(torch.allclose(out_input, out_state))


if __name__ == "__main__":
    unittest.main()
