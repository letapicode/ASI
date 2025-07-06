import torch
from typing import Callable

from .moe_router import SwitchRouter


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


__all__ = ["RLMoERouter"]
