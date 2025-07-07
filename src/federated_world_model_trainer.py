from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from .world_model_rl import RLBridgeConfig, WorldModel, TransitionDataset
from .secure_federated_learner import SecureFederatedLearner
from .zk_gradient_proof import ZKGradientProof

@dataclass
class FederatedTrainerConfig:
    rounds: int = 1
    local_epochs: int = 1
    lr: float = 1e-4

class FederatedWorldModelTrainer:
    """Train a world model across nodes via secure gradient averaging."""

    def __init__(
        self,
        cfg: RLBridgeConfig,
        datasets: Iterable[Dataset],
        learner: SecureFederatedLearner | None = None,
        trainer_cfg: FederatedTrainerConfig | None = None,
        reward_sync_hook: Callable[[], Iterable[Iterable[float]]] | None = None,
    ) -> None:
        """Initialize the trainer.

        Parameters
        ----------
        cfg:
            Base configuration for the world model.
        datasets:
            Training data for each participating node.
        learner:
            Encryption helper for federated averaging.
        trainer_cfg:
            Hyper-parameters controlling the number of rounds and optimizer.
        reward_sync_hook:
            Optional callable returning updated rewards for each dataset at the
            start of every round.
        """

        self.cfg = cfg
        self.datasets = list(datasets)
        self.learner = learner or SecureFederatedLearner()
        self.tcfg = trainer_cfg or FederatedTrainerConfig()
        self.model = WorldModel(cfg)
        self.reward_sync_hook = reward_sync_hook

    # --------------------------------------------------
    def _local_gradients(self, dataset: Dataset) -> List[torch.Tensor]:
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = [torch.zeros_like(p) for p in params]
        opt = torch.optim.SGD(params, lr=self.tcfg.lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(self.tcfg.local_epochs):
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
        for _ in range(self.tcfg.rounds):
            if self.reward_sync_hook is not None:
                rewards = self.reward_sync_hook()
                for ds, rs in zip(self.datasets, rewards, strict=True):
                    if isinstance(ds, TransitionDataset):
                        ds.data = [
                            (s, a, ns, float(r))
                            for (s, a, ns, _), r in zip(ds.data, rs, strict=True)
                        ]
            enc_grads: List[Tuple[torch.Tensor, ZKGradientProof | None]] = []
            for ds in self.datasets:
                grads = self._local_gradients(ds)
                flat = torch.cat([g.view(-1) for g in grads])
                if self.learner.require_proof:
                    enc, proof = self.learner.encrypt(flat, with_proof=True)
                else:
                    enc = self.learner.encrypt(flat)
                    proof = None
                enc_grads.append((enc, proof))
            agg = self.learner.aggregate(
                [self.learner.decrypt(g, p) for g, p in enc_grads],
                proofs=[pr.digest for _, pr in enc_grads if pr is not None] if self.learner.require_proof else None,
            )
            start = 0
            for p in params:
                num = p.numel()
                g = agg[start : start + num].view_as(p)
                p.data -= self.tcfg.lr * g
                start += num
        return self.model

__all__ = ["FederatedWorldModelTrainer", "FederatedTrainerConfig"]
