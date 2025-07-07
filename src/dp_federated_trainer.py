from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable, List

import torch
from torch.utils.data import DataLoader, Dataset

from .world_model_rl import RLBridgeConfig, WorldModel, TransitionDataset
from .secure_federated_learner import SecureFederatedLearner
from .differential_privacy_optimizer import (
    DifferentialPrivacyOptimizer,
    DifferentialPrivacyConfig,
)


@dataclass
class DPFederatedTrainerConfig:
    """Configuration for federated training with differential privacy."""

    rounds: int = 1
    local_epochs: int = 1
    lr: float = 1e-4
    clip_norm: float = 1.0
    noise_std: float = 0.01


class DPFederatedTrainer:
    """Federated world-model trainer that applies differential privacy."""

    def __init__(
        self,
        cfg: RLBridgeConfig,
        datasets: Iterable[Dataset],
        learner: SecureFederatedLearner | None = None,
        dp_cfg: DPFederatedTrainerConfig | None = None,
        reward_sync_hook: Callable[[], Iterable[Iterable[float]]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.datasets = list(datasets)
        self.learner = learner or SecureFederatedLearner()
        self.dp_cfg = dp_cfg or DPFederatedTrainerConfig()
        self.model = WorldModel(cfg)
        self.reward_sync_hook = reward_sync_hook
        dp_opt_cfg = DifferentialPrivacyConfig(
            lr=self.dp_cfg.lr,
            clip_norm=self.dp_cfg.clip_norm,
            noise_std=self.dp_cfg.noise_std,
        )
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.dp_opt = DifferentialPrivacyOptimizer(params, dp_opt_cfg)

    # --------------------------------------------------
    def _local_gradients(self, dataset: Dataset) -> List[torch.Tensor]:
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = [torch.zeros_like(p) for p in params]
        opt = torch.optim.SGD(params, lr=self.dp_cfg.lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(self.dp_cfg.local_epochs):
            for s, a, ns, r in loader:
                pred_s, pred_r = self.model(s, a)
                loss = loss_fn(pred_s, ns) + loss_fn(pred_r, r)
                opt.zero_grad()
                loss.backward()
                for g, p in zip(grads, params):
                    g += p.grad.detach().clone()
                opt.step()
        return [g / len(loader) for g in grads]

    def train(self) -> WorldModel:
        params = [p for p in self.model.parameters() if p.requires_grad]
        for _ in range(self.dp_cfg.rounds):
            if self.reward_sync_hook is not None:
                rewards = self.reward_sync_hook()
                for ds, rs in zip(self.datasets, rewards, strict=True):
                    if isinstance(ds, TransitionDataset):
                        ds.data = [
                            (s, a, ns, float(r))
                            for (s, a, ns, _), r in zip(ds.data, rs, strict=True)
                        ]
            enc_grads = []
            for ds in self.datasets:
                grads = self._local_gradients(ds)
                flat = torch.cat([g.view(-1) for g in grads])
                enc = self.learner.encrypt(flat)
                enc_grads.append(enc)
            agg = self.learner.aggregate([self.learner.decrypt(g) for g in enc_grads])
            self.dp_opt.zero_grad()
            start = 0
            for p in params:
                num = p.numel()
                g = agg[start : start + num].view_as(p)
                p.grad = g.clone()
                start += num
            self.dp_opt.step()
        return self.model


__all__ = ["DPFederatedTrainer", "DPFederatedTrainerConfig"]
