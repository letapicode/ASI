import torch
from torch import nn

class RetNetRetention(nn.Module):
    """Simplified retention module for C-1 experiments."""

    def __init__(self, decay: float = 0.9) -> None:
        super().__init__()
        self.decay = decay

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute RetNet-style retention.

        Args:
            q: Query tensor of shape (batch, seq, dim).
            k: Key tensor of shape (batch, seq, dim).
            v: Value tensor of shape (batch, seq, dim).
        Returns:
            Tensor of shape (batch, seq, dim).
        """
        batch, seq, dim = q.shape
        r = torch.zeros(batch, dim, device=q.device, dtype=q.dtype)
        outputs = []
        for t in range(seq):
            r = self.decay * r + k[:, t] * v[:, t]
            outputs.append(q[:, t] * r)
        return torch.stack(outputs, dim=1)
