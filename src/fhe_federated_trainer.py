from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch

try:
    import tenseal as ts  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ts = None

from .secure_federated_learner import SecureFederatedLearner
from .proofs import ZKGradientProof
from .world_model_rl import RLBridgeConfig, WorldModel, TransitionDataset
from .fhe_runner import run_fhe


@dataclass
class FHEFederatedTrainerConfig:
    rounds: int = 1
    local_epochs: int = 1
    lr: float = 1e-4


class FHEFederatedTrainer:
    """Federated trainer that encrypts gradients via TenSEAL."""

    def __init__(
        self,
        cfg: RLBridgeConfig,
        datasets: Iterable[TransitionDataset],
        ctx: "ts.Context",
        learner: SecureFederatedLearner | None = None,
        trainer_cfg: FHEFederatedTrainerConfig | None = None,
    ) -> None:
        if ts is None:
            raise ImportError("tenseal is required for FHEFederatedTrainer")
        self.cfg = cfg
        self.datasets = list(datasets)
        self.ctx = ctx
        self.learner = learner or SecureFederatedLearner()
        self.tcfg = trainer_cfg or FHEFederatedTrainerConfig()
        self.model = WorldModel(cfg)

    # --------------------------------------------------
    def _local_gradients(self, dataset: TransitionDataset) -> List[torch.Tensor]:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True
        )
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

    def _decrypt_fhe(self, enc: "ts.CKKSVector") -> torch.Tensor:
        """Decrypt an encrypted vector using ``run_fhe``."""

        def _model(_: "ts.CKKSVector", vec: "ts.CKKSVector" = enc) -> "ts.CKKSVector":
            return vec

        dummy = torch.zeros(enc.size())
        return run_fhe(_model, dummy, self.ctx)

    def train(self) -> WorldModel:
        params = [p for p in self.model.parameters() if p.requires_grad]
        for _ in range(self.tcfg.rounds):
            enc_grads: List[Tuple["ts.CKKSVector", ZKGradientProof]] = []
            for ds in self.datasets:
                grads = self._local_gradients(ds)
                flat = torch.cat([g.view(-1) for g in grads])
                enc, proof = self.learner.encrypt(flat, with_proof=True)
                enc_vec = ts.ckks_vector(self.ctx, enc.tolist())
                enc_grads.append((enc_vec, proof))
            dec_grads = [
                self.learner.decrypt(self._decrypt_fhe(g), proof)
                for g, proof in enc_grads
            ]
            agg = self.learner.aggregate(dec_grads)
            start = 0
            for p in params:
                num = p.numel()
                g = agg[start : start + num].view_as(p)
                p.data -= self.tcfg.lr * g
                start += num
        return self.model


__all__ = ["FHEFederatedTrainer", "FHEFederatedTrainerConfig"]
