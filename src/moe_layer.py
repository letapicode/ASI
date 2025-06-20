import torch
from torch import nn
from .moe_router import HashRouter, SwitchRouter


class MoELayer(nn.Module):
    """Minimal Mixture-of-Experts feed-forward block implementing S-1 routing."""

    def __init__(self, dim: int, hidden: int, num_experts: int, router: str = "hash", k: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.num_experts = num_experts
        self.k = k
        if router == "switch":
            self.router = SwitchRouter(dim=dim, num_experts=num_experts, k=k)
        else:
            self.router = HashRouter(num_experts=num_experts, k=k)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
                                      for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to experts and combine their outputs."""
        assign = self.router(x)
        batch, seq, _ = x.shape
        out = torch.zeros(batch, seq, self.dim, device=x.device, dtype=x.dtype)
        flat_x = x.reshape(batch * seq, self.dim)
        flat_out = out.reshape(batch * seq, self.dim)
        for idx, expert in enumerate(self.experts):
            mask = (assign == idx).any(-1).view(-1)
            if mask.any():
                out_tokens = expert(flat_x[mask])
                flat_out[mask] += out_tokens
        out = flat_out.view(batch, seq, self.dim)
        return out / self.k
