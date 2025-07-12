import torch
from torch import nn


class RetNetRetention(nn.Module):
    """Simplified retention module supporting multiple heads."""

    def __init__(self, num_heads: int = 1, decay: float | list[float] = 0.9) -> None:
        super().__init__()
        self.num_heads = num_heads
        if isinstance(decay, float):
            decay = [decay] * num_heads
        if len(decay) != num_heads:
            raise ValueError("decay must have length equal to num_heads")
        self.register_buffer("decay", torch.tensor(decay).view(1, num_heads, 1))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute RetNet-style retention for multiple heads."""
        batch, seq, dim = q.shape
        if dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        head_dim = dim // self.num_heads

        q = q.view(batch, seq, self.num_heads, head_dim)
        k = k.view(batch, seq, self.num_heads, head_dim)
        v = v.view(batch, seq, self.num_heads, head_dim)

        r = torch.zeros(batch, self.num_heads, head_dim, device=q.device, dtype=q.dtype)
        outputs = []
        for t in range(seq):
            r = self.decay * r + k[:, t] * v[:, t]
            outputs.append(q[:, t] * r)
        out = torch.stack(outputs, dim=1)
        return out.view(batch, seq, dim)


class HybridRetention(nn.Module):
    """RetNet decay kernel followed by a Mamba-style linear update."""

    def __init__(
        self,
        num_heads: int,
        dim: int,
        decay: float | list[float] = 0.9,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        if isinstance(decay, float):
            decay = [decay] * num_heads
        if len(decay) != num_heads:
            raise ValueError("decay must have length equal to num_heads")
        self.register_buffer("decay", torch.tensor(decay).view(1, num_heads, 1))
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.A = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.B = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.drop = nn.Dropout(dropout)
        self.use_residual = residual

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply retention then gated linear recurrence."""
        batch, seq, dim = q.shape
        if dim != self.dim:
            raise ValueError("dim mismatch")
        if dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        head_dim = dim // self.num_heads

        q = q.view(batch, seq, self.num_heads, head_dim)
        k = k.view(batch, seq, self.num_heads, head_dim)
        v = v.view(batch, seq, self.num_heads, head_dim)

        r = torch.zeros(batch, self.num_heads, head_dim, device=q.device, dtype=q.dtype)
        state = torch.zeros(batch, dim, device=q.device, dtype=q.dtype)
        outputs = []
        for t in range(seq):
            r = self.decay * r + k[:, t] * v[:, t]
            retained = (q[:, t] * r).view(batch, dim)

            inp = self.in_proj(retained)
            gate = torch.sigmoid(self.gate(inp))
            blended = gate * state + (1 - gate) * inp
            state = torch.tanh(blended @ self.A.t() + inp @ self.B.t())
            state = self.drop(state)
            out = self.out_proj(state)
            if self.use_residual:
                out = out + retained
            outputs.append(out)
        return torch.stack(outputs, dim=1)


__all__ = ["RetNetRetention", "HybridRetention"]
