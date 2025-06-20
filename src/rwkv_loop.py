import torch
from torch import nn


class RWKVLoop(nn.Module):
    """Simplified RWKV-style recurrent block for Plan.md C-6."""

    def __init__(self, dim: int, decay: float = 0.25, mix: float = 0.5) -> None:
        super().__init__()
        self.dim = dim
        self.decay = nn.Parameter(torch.tensor(decay))
        self.mix = nn.Parameter(torch.tensor(mix))
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a sequence with constant-memory recurrence.

        Args:
            x: Tensor of shape ``(batch, seq, dim)``.
        Returns:
            Tensor with the same shape as ``x``.
        """
        batch, seq, dim = x.shape
        state = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)
        shifted = torch.cat([
            torch.zeros(batch, 1, dim, device=x.device, dtype=x.dtype),
            x[:, :-1],
        ], dim=1)
        outputs = []
        for t in range(seq):
            inp = self.mix * x[:, t] + (1 - self.mix) * shifted[:, t]
            inp = torch.relu(self.in_proj(inp))
            state = state * self.decay + inp
            outputs.append(self.out_proj(state))
        return torch.stack(outputs, dim=1)
