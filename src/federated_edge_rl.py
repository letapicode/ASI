from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch

from .edge_rl_trainer import EdgeRLTrainer
from .secure_federated_learner import SecureFederatedLearner
from .differential_privacy_optimizer import (
    DifferentialPrivacyConfig,
    DifferentialPrivacyOptimizer,
)
from .compute_budget_tracker import ComputeBudgetTracker


@dataclass
class FederatedEdgeConfig:
    rounds: int = 1
    local_steps: int = 1
    lr: float = 1e-3


class FederatedEdgeRLTrainer:
    """Federated wrapper around :class:`EdgeRLTrainer`."""

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: FederatedEdgeConfig | None = None,
        learner: SecureFederatedLearner | None = None,
        dp_cfg: DifferentialPrivacyConfig | None = None,
        budget: ComputeBudgetTracker | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or FederatedEdgeConfig()
        self.learner = learner or SecureFederatedLearner()
        self.dp_cfg = dp_cfg or DifferentialPrivacyConfig(lr=self.cfg.lr)
        self.budget = budget or ComputeBudgetTracker(float("inf"))

    # --------------------------------------------------
    def _local_gradients(
        self, data: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[torch.Tensor]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in params]
        opt: torch.optim.Optimizer
        if isinstance(self.dp_cfg, DifferentialPrivacyConfig):
            opt = DifferentialPrivacyOptimizer(params, self.dp_cfg)
        else:
            opt = torch.optim.SGD(params, lr=self.cfg.lr)
        trainer = EdgeRLTrainer(self.model, opt, self.budget)
        subset = list(data)[: self.cfg.local_steps]
        steps = trainer.train(subset, threshold=0.0)
        after = [p.detach().clone() for p in params]
        # Restore original weights
        for p, b in zip(params, before):
            p.data.copy_(b)
        if steps == 0:
            return [torch.zeros_like(b) for b in before]
        grads = [-(a - b) / (self.cfg.lr * steps) for a, b in zip(after, before)]
        # Clip and add noise similar to DifferentialPrivacyOptimizer
        total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
        clip_coef = min(1.0, self.dp_cfg.clip_norm / (total_norm + 1e-6))
        grads = [g * clip_coef for g in grads]
        grads = [g + torch.randn_like(g) * self.dp_cfg.noise_std for g in grads]
        return grads

    # --------------------------------------------------
    def train(
        self, datasets: Iterable[Iterable[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> torch.nn.Module:
        params = [p for p in self.model.parameters() if p.requires_grad]
        for _ in range(self.cfg.rounds):
            enc_grads = []
            for ds in datasets:
                grads = self._local_gradients(ds)
                flat = torch.cat([g.view(-1) for g in grads])
                enc = self.learner.encrypt(flat)
                enc_grads.append(enc)
            agg = self.learner.aggregate([self.learner.decrypt(g) for g in enc_grads])
            start = 0
            for p in params:
                num = p.numel()
                g = agg[start : start + num].view_as(p)
                p.data -= self.cfg.lr * g
                start += num
        return self.model


__all__ = ["FederatedEdgeRLTrainer", "FederatedEdgeConfig"]
