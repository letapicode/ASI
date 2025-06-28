import torch
from torch import nn


class MambaBlock(nn.Module):
    """Simplified state-space block for Plan.md C-2."""

    def __init__(self, dim: int, dropout: float = 0.0, residual: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.A = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.B = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.use_residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a sequence with recurrent state updates.

        Args:
            x: Tensor of shape (batch, seq, dim).
        Returns:
            Tensor with the same shape.
        """
        batch, seq, dim = x.shape
        state = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq):
            inp = self.in_proj(x[:, t])
            gate = torch.sigmoid(self.gate(inp))
            blended = gate * state + (1 - gate) * inp
            state = torch.tanh(blended @ self.A.t() + inp @ self.B.t())
            state = self.drop(state)
            out = self.out_proj(state)
            if self.use_residual:
                out = out + x[:, t]
            outputs.append(out)
        return torch.stack(outputs, dim=1)
