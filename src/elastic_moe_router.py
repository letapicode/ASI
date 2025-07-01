import torch
from typing import Callable

from .moe_router import SwitchRouter


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
