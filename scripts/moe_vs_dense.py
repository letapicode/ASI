import torch
from torch import nn
from src.moe_router import HashRouter, SwitchRouter
from src.elastic_moe_router import ElasticMoERouter
import argparse

class ToyModel(nn.Module):
    """Minimal feed-forward model with optional MOE."""
    def __init__(self, dim: int, hidden: int, num_experts: int = 0, router_type: str = "hash") -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.moe = num_experts > 0
        if self.moe:
            if router_type == "switch":
                self.router = SwitchRouter(dim=dim, num_experts=num_experts)
            elif router_type == "elastic":
                self.router = ElasticMoERouter(dim=dim, num_experts=num_experts)
            else:
                self.router = HashRouter(num_experts)
            self.experts = nn.ModuleList([
                nn.Linear(dim, hidden) for _ in range(num_experts)
            ])
        else:
            self.fc = nn.Linear(dim, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.moe:
            assign = self.router(x)
            batch, seq, _ = x.shape
            hidden = torch.zeros(batch, seq, self.hidden, device=x.device)
            for i, layer in enumerate(self.experts):
                mask = (assign == i).any(-1)
                if mask.any():
                    hidden[mask] = layer(x[mask])
        else:
            hidden = self.fc(x)
        return self.out(torch.relu(hidden))

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def approx_flops(model: nn.Module, tokens: int) -> int:
    # Rough estimate: 2 * params * tokens
    return 2 * param_count(model) * tokens

def build_and_profile(dim: int, hidden: int, tokens: int, num_experts: int = 0, router_type: str = "hash"):
    x = torch.randn(1, tokens, dim)
    model = ToyModel(dim, hidden, num_experts, router_type)
    flops = approx_flops(model, tokens)
    out = model(x)
    out.sum().backward()
    return param_count(model), flops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare dense and MOE models")
    parser.add_argument(
        "--router",
        choices=["hash", "switch", "elastic"],
        default="hash",
        help="Router type for MOE",
    )
    parser.add_argument("--experts", type=int, default=16, help="Number of experts")
    args = parser.parse_args()

    dense_params, dense_flops = build_and_profile(256, 512, 1024, 0)
    moe_params, moe_flops = build_and_profile(256, 512, 1024, args.experts, args.router)
    print("Dense params:", dense_params)
    print(f"MOE params ({args.router}):", moe_params)
    print("Param ratio:", moe_params / dense_params)
    print("FLOP ratio:", moe_flops / dense_flops)
