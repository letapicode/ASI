import torch
from torch import nn
from .moe_router import BaseRouter, HashRouter, SwitchRouter, balance_loss
from .elastic_moe_router import ElasticMoERouter


class MoELayer(nn.Module):
    """Minimal Mixture-of-Experts feed-forward block implementing S-1 routing."""

    def __init__(self, dim: int, hidden: int, num_experts: int,
                 router: str | BaseRouter = "hash", k: int = 2,
                 balance_weight: float | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.num_experts = num_experts
        self.k = k
        self.balance_weight = balance_weight
        if isinstance(router, BaseRouter):
            self.router = router
        elif router == "switch":
            self.router = SwitchRouter(dim=dim, num_experts=num_experts, k=k)
        elif router == "elastic":
            self.router = ElasticMoERouter(dim=dim, num_experts=num_experts, k=k)
        else:
            self.router = HashRouter(num_experts=num_experts, k=k)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
                                      for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        """Route tokens to experts and combine their outputs."""
        result = self.router(x)
        if isinstance(result, tuple):
            assign, weights = result
        else:
            assign = result
            weights = torch.full(assign.shape, 1.0 / self.k, device=x.device, dtype=x.dtype)

        batch, seq, _ = x.shape
        out = torch.zeros(batch, seq, self.dim, device=x.device, dtype=x.dtype)
        flat_x = x.reshape(batch * seq, self.dim)
        flat_out = out.reshape(batch * seq, self.dim)
        flat_assign = assign.reshape(batch * seq, self.k)
        flat_weights = weights.reshape(batch * seq, self.k)
        for idx, expert in enumerate(self.experts):
            token_w = (flat_weights * (flat_assign == idx)).sum(-1)
            mask = token_w > 0
            if mask.any():
                out_tokens = expert(flat_x[mask]) * token_w[mask].unsqueeze(-1)
                flat_out[mask] += out_tokens
        out = flat_out.view(batch, seq, self.dim)

        if self.balance_weight:
            penalty = self.balance_weight * balance_loss(assign, self.num_experts)
            return out, penalty
        return out
