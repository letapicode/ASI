import torch
from torch import nn

class HybridRetention(nn.Module):
    """Combine Mamba-style linear updates with RetNet decay."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        decay: float | list[float] = 0.9,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if isinstance(decay, float):
            decay = [decay] * num_heads
        if len(decay) != num_heads:
            raise ValueError("decay must match num_heads")
        self.register_buffer("decay", torch.tensor(decay).view(1, num_heads, 1))
        self.A = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.B = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.gate = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return sequence processed by hybrid retention."""
        batch, seq, dim = q.shape
        if dim != self.dim or dim % self.num_heads != 0:
            raise ValueError("dimension mismatch")
        head_dim = dim // self.num_heads

        q = q.view(batch, seq, self.num_heads, head_dim)
        k = k.view(batch, seq, self.num_heads, head_dim)
        v = v.view(batch, seq, self.num_heads, head_dim)

        r = torch.zeros(batch, self.num_heads, head_dim, device=q.device, dtype=q.dtype)
        state = torch.zeros(batch, self.num_heads, head_dim, device=q.device, dtype=q.dtype)
        outputs = []
        for t in range(seq):
            r = self.decay * r + k[:, t] * v[:, t]
            inp = q[:, t]
            gate = torch.sigmoid(self.gate(inp.view(batch, -1))).view(batch, self.num_heads, head_dim)
            blended = gate * state + (1 - gate) * inp
            s = torch.tanh(
                blended.view(batch, dim) @ self.A.t()
                + r.view(batch, dim) @ self.B.t()
            )
            s = self.drop(s).view(batch, self.num_heads, head_dim)
            state = s
            out = self.out_proj(state.view(batch, dim)).view(batch, self.num_heads, head_dim)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        return out.view(batch, seq, dim)

__all__ = ["HybridRetention"]
