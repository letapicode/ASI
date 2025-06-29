import torch
from torch import nn
from abc import ABC, abstractmethod


class BaseRouter(nn.Module, ABC):
    """Abstract base router specifying the routing interface."""

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Return expert assignments or (assignments, weights) for ``x``."""

    @abstractmethod
    def load_balance_std(self, assignments: torch.Tensor) -> float:
        """Compute relative standard deviation of expert loads."""

class HashRouter(BaseRouter):
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

    def expert_utilization(self, assignments: torch.Tensor) -> torch.Tensor:
        """Return number of tokens routed to each expert."""
        return torch.bincount(assignments.view(-1), minlength=self.num_experts)


class SwitchRouter(BaseRouter):
    """Top-k gating network for S-1 sparse Mixture-of-Experts."""

    def __init__(self, dim: int, num_experts: int, k: int = 2,
                 temperature: float = 1.0) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.temperature = temperature
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor):
        """Select top-k experts via learned gating.

        Args:
            x: Tensor of shape (batch, seq, dim).
        Returns:
            Tuple of tensors ``(assignments, weights)`` each with shape
            ``(batch, seq, k)`` where ``assignments`` holds the expert indices
            and ``weights`` the corresponding normalized gate values.
        """
        logits = self.gate(x) / self.temperature
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, self.k, dim=-1)
        return topk.indices, topk.values

    def load_balance_std(self, assignments: torch.Tensor) -> float:
        counts = torch.bincount(assignments.view(-1), minlength=self.num_experts).float()
        return (counts.std() / counts.mean()).item()

    def expert_utilization(self, assignments: torch.Tensor) -> torch.Tensor:
        return torch.bincount(assignments.view(-1), minlength=self.num_experts)


def balance_loss(assignments: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Return relative std of expert usage as a loss term."""
    counts = torch.bincount(assignments.view(-1), minlength=num_experts).float()
    mean = counts.mean()
    if mean == 0:
        return torch.tensor(0.0, device=assignments.device)
    return counts.std() / mean


def balance_loss_probs(probs: torch.Tensor) -> torch.Tensor:
    """Differentiable load-balance penalty from routing probabilities."""
    counts = probs.sum(dim=(-3, -2))
    mean = counts.mean()
    if mean == 0:
        return torch.tensor(0.0, device=probs.device)
    return counts.std() / mean


def token_drop_rate(assignments: torch.Tensor, num_experts: int, capacity: int) -> float:
    """Return fraction of tokens exceeding per-expert ``capacity``."""
    counts = torch.bincount(assignments.view(-1), minlength=num_experts).float()
    dropped = torch.clamp(counts - capacity, min=0).sum()
    return (dropped / assignments.numel()).item()
