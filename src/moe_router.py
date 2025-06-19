import torch
from torch import nn

class HashRouter(nn.Module):
    """O(1) router for Sparse Mixture-of-Experts (Plan.md S-1)."""

    def __init__(self, num_experts: int, k: int = 2, seed: int = 42) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.seed = seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to experts with hash-based gating.

        Args:
            x: Tensor of shape (batch, seq, dim).
        Returns:
            Tensor of shape (batch * seq, k) with expert indices.
        """
        batch, seq, _ = x.shape
        flat_indices = torch.arange(batch * seq, device=x.device)
        hashed = (flat_indices * 0x9E3779B97F4A7C15 + self.seed) % self.num_experts
        assignments = [(hashed + offset) % self.num_experts for offset in range(self.k)]
        return torch.stack(assignments, dim=-1)

    def load_balance_std(self, assignments: torch.Tensor) -> float:
        """Compute relative standard deviation of expert loads."""
        counts = torch.bincount(assignments.view(-1), minlength=self.num_experts).float()
        return (counts.std() / counts.mean()).item()
