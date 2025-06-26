import unittest
import torch
import torch.nn.functional as F

from asi.hyena_filter import HyenaFilter


class TestHyenaFilter(unittest.TestCase):
    def test_fft_conv_matches_conv1d(self):
        filt_len = 4
        module = HyenaFilter(filter_length=filt_len)
        module.filter.data = torch.randn(filt_len)
        x = torch.randn(2, 8, 3)
        out = module(x)
        # reference using conv1d
        ref = F.conv1d(
            x.transpose(1, 2).reshape(-1, 1, x.shape[1]),
            module.filter.flip(0).view(1, 1, -1),
            padding=filt_len - 1,
        ).reshape(x.shape[0], x.shape[2], -1)[:, :, : x.shape[1]].transpose(1, 2)
        self.assertTrue(torch.allclose(out, ref, atol=1e-5, rtol=1e-5))
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
