import torch
from torch import nn

from .mamba_block import MambaBlock
from .retnet_retention import RetNetRetention

class HybridRetention(nn.Module):
    """Combine Mamba-style linear state update with RetNet decay kernel."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        decay: float | list[float] = 0.9,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.mamba = MambaBlock(dim=dim, dropout=dropout, residual=residual)
        self.retention = RetNetRetention(num_heads=num_heads, decay=decay)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply a MambaBlock to ``q`` then RetNet-style retention."""
        q_proj = self.mamba(q)
        return self.retention(q_proj, k, v)


__all__ = ["HybridRetention"]
