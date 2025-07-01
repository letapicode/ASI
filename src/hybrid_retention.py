import torch
from torch import nn

from .retnet_retention import RetNetRetention
from .mamba_block import MambaBlock


class HybridRetention(nn.Module):
    """Fuse MambaBlock updates with RetNet-style retention."""

    def __init__(self, dim: int, num_heads: int = 1, decay: float | list[float] = 0.9, dropout: float = 0.0) -> None:
        super().__init__()
        self.mamba = MambaBlock(dim=dim, dropout=dropout, residual=False)
        self.retention = RetNetRetention(num_heads=num_heads, decay=decay)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return combined Mamba and RetNet outputs."""
        m_out = self.mamba(q)
        r_out = self.retention(q, k, v)
        return m_out + r_out


__all__ = ["HybridRetention"]
