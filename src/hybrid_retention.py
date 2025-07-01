import torch
from torch import nn

from .mamba_block import MambaBlock
from .retnet_retention import RetNetRetention


class HybridRetention(nn.Module):
    """Fuse RetNet-style retention with a Mamba linear update."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        decay: float | list[float] = 0.9,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.retention = RetNetRetention(num_heads=num_heads, decay=decay)
        self.mamba = MambaBlock(dim=dim, dropout=dropout, residual=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply retention then a gated linear recurrence."""
        retained = self.retention(q, k, v)
        return self.mamba(retained)


__all__ = ["HybridRetention"]
