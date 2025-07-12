import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Callable


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


class ElasticMoERouter(SwitchRouter):
    """Switch router that reduces active experts when GPU utilization is high."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        k: int = 2,
        temperature: float = 1.0,
        min_util: float = 0.5,
        max_util: float = 0.9,
        utilization_fn: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(dim=dim, num_experts=num_experts, k=k, temperature=temperature)
        self.min_util = min_util
        self.max_util = max_util
        self.utilization_fn = utilization_fn
        self.active_experts = num_experts

    def gpu_utilization(self) -> float:
        """Return current GPU memory utilization ratio."""
        if self.utilization_fn is not None:
            return float(self.utilization_fn())
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return float(alloc) / float(total)
        return 0.0

    def _adjust_active_experts(self) -> None:
        util = self.gpu_utilization()
        if util >= self.max_util:
            scale = 0.25
        elif util >= self.min_util:
            scale = 0.5
        else:
            scale = 1.0
        self.active_experts = max(1, int(self.num_experts * scale))

    def forward(self, x: torch.Tensor):
        self._adjust_active_experts()
        assignments, weights = super().forward(x)
        if self.active_experts < self.num_experts:
            assignments = assignments % self.active_experts
        return assignments, weights

    def load_balance_std(self, assignments: torch.Tensor) -> float:
        counts = torch.bincount(assignments.view(-1), minlength=self.active_experts).float()
        return (counts.std() / counts.mean()).item()


class RLMoERouter(SwitchRouter):
    """Switch router that learns routing probabilities with REINFORCE."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        k: int = 2,
        temperature: float = 1.0,
        min_util: float = 0.5,
        max_util: float = 0.9,
        utilization_fn: Callable[[], float] | None = None,
        lr: float = 0.1,
    ) -> None:
        super().__init__(dim=dim, num_experts=num_experts, k=k, temperature=temperature)
        self.min_util = min_util
        self.max_util = max_util
        self.utilization_fn = utilization_fn
        self.lr = lr
        self.baseline = 0.0
        self.prefs = torch.zeros(num_experts, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        probs = torch.softmax(self.prefs / self.temperature, dim=0)
        batch, seq, _ = x.shape
        flat_probs = probs.repeat(batch * seq, 1)
        flat_assign = torch.multinomial(flat_probs, self.k, replacement=True)
        assignments = flat_assign.view(batch, seq, self.k)
        weights = probs[assignments]

        reward = -self.load_balance_std(assignments)
        adv = reward - self.baseline
        self.baseline += self.lr * adv
        counts = torch.bincount(assignments.view(-1), minlength=self.num_experts).float()
        freq = counts / counts.sum()
        for i in range(self.num_experts):
            grad = adv * (freq[i] - probs[i])
            self.prefs[i] += self.lr * grad

        return assignments, weights

    def load_balance_std(self, assignments: torch.Tensor) -> float:
        counts = torch.bincount(assignments.view(-1), minlength=self.num_experts).float()
        return (counts.std() / counts.mean()).item()


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


__all__ = [
    "BaseRouter",
    "HashRouter",
    "SwitchRouter",
    "ElasticMoERouter",
    "RLMoERouter",
    "balance_loss",
    "balance_loss_probs",
    "token_drop_rate",
]
