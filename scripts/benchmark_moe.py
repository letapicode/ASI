import torch
from torch import nn
from src.moe_router import HashRouter, SwitchRouter
import argparse

# Simple feed-forward network for demonstration
class ToyModel(nn.Module):
    def __init__(self, dim: int, hidden: int, num_experts: int = 0, router_type: str = "hash"):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.moe = num_experts > 0
        if self.moe:
            if router_type == "switch":
                self.router = SwitchRouter(dim=dim, num_experts=num_experts)
            else:
                self.router = HashRouter(num_experts)
            self.experts = nn.ModuleList([
                nn.Linear(dim, hidden) for _ in range(num_experts)
            ])
        else:
            self.fc = nn.Linear(dim, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x):
        if self.moe:
            assign = self.router(x)
            batch, seq, _ = x.shape
            expert_out = torch.zeros(batch, seq, self.hidden, device=x.device)
            for idx, layer in enumerate(self.experts):
                mask = (assign == idx).any(-1)
                if mask.any():
                    expert_out[mask] = layer(x[mask])
        else:
            expert_out = self.fc(x)
        return self.out(torch.relu(expert_out))

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def approx_flops(model: nn.Module, tokens: int) -> int:
    # extremely rough estimation: 2 * param_count per token
    return 2 * count_parameters(model) * tokens

def run(dim=256, hidden=512, tokens=1024, num_experts=0, router_type="hash"):
    x = torch.randn(1, tokens, dim)
    model = ToyModel(dim, hidden, num_experts, router_type)
    flops = approx_flops(model, tokens)
    out = model(x)
    out.sum().backward()
    return count_parameters(model), flops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MOE vs dense model")
    parser.add_argument("--router", choices=["hash", "switch"], default="hash", help="Router type for MOE")
    parser.add_argument("--experts", type=int, default=16, help="Number of experts")
    args = parser.parse_args()

    dense_params, dense_flops = run(num_experts=0)
    moe_params, moe_flops = run(num_experts=args.experts, router_type=args.router)
    print("Dense params:", dense_params)
    print(f"MOE params ({args.router}):", moe_params)
    print("Param increase:", moe_params / dense_params)
    print("FLOP ratio:", moe_flops / dense_flops)
