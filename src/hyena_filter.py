import torch
from torch import nn


class HyenaFilter(nn.Module):
    """FFT-based 1D convolution for Plan.md C-3."""

    def __init__(self, filter_length: int) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.filter = nn.Parameter(torch.randn(filter_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learnable filter along the sequence dimension.

        Args:
            x: Tensor of shape (batch, seq, dim).
        Returns:
            Tensor with the same shape as ``x``.
        """
        batch, seq, dim = x.shape
        if seq <= 64:
            weight = self.filter.flip(0).view(1, 1, -1).repeat(dim, 1, 1)
            conv = torch.nn.functional.conv1d(
                x.transpose(1, 2), weight, padding=self.filter_length - 1, groups=dim
            )
            return conv[:, :, :seq].transpose(1, 2)
        pad = seq + self.filter_length - 1
        fx = torch.fft.rfft(x.transpose(1, 2), n=pad)
        ff = torch.fft.rfft(self.filter.flip(0), n=pad)
        conv = torch.fft.irfft(fx * ff, n=pad)
        conv = conv[:, :, self.filter_length - 1 : self.filter_length - 1 + seq]
        return conv.transpose(1, 2)
