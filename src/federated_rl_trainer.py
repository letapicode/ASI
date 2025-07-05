from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .self_play_env import SimpleEnv, rollout_env
from .self_play_skill_loop import SelfPlaySkillLoopConfig
from .secure_federated_learner import SecureFederatedLearner
from .zk_verifier import ZKVerifier


@dataclass
class FederatedRLTrainerConfig:
    """Configuration for federated self-play training."""

    rounds: int = 1
    local_steps: int = 5
    lr: float = 1e-3


class _TrajectoryDataset(Dataset):
    def __init__(self, states: Iterable[torch.Tensor], actions: Iterable[int]):
        self.states = list(states)
        self.actions = torch.tensor(list(actions), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return self.states[idx], self.actions[idx]


class PolicyNet(nn.Module):
    """Tiny policy network for vector observations."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - pass through
        return self.net(x)


class FederatedRLTrainer:
    """Run self-play locally and aggregate gradients via a secure learner."""

    def __init__(
        self,
        sp_cfg: SelfPlaySkillLoopConfig,
        learner: SecureFederatedLearner | None = None,
        frl_cfg: FederatedRLTrainerConfig | None = None,
        zk: ZKVerifier | None = None,
    ) -> None:
        self.sp_cfg = sp_cfg
        self.cfg = frl_cfg or FederatedRLTrainerConfig()
        self.learner = learner or SecureFederatedLearner()
        self.zk = zk or ZKVerifier()
        self.policy = PolicyNet(sp_cfg.env_state_dim, sp_cfg.action_dim)

    # --------------------------------------------------
    def _policy_act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.policy(obs.unsqueeze(0))
        return logits.argmax(dim=-1).squeeze(0)

    def _collect_experience(
        self, env: SimpleEnv
    ) -> Tuple[List[torch.Tensor], List[int]]:
        states, _, actions = rollout_env(
            env, self._policy_act, steps=self.cfg.local_steps, return_actions=True
        )
        return states, actions

    def _local_gradients(
        self, states: Iterable[torch.Tensor], actions: Iterable[int]
    ) -> List[torch.Tensor]:
        dataset = _TrajectoryDataset(states, actions)
        loader = DataLoader(dataset, batch_size=len(dataset))
        params = [p for p in self.policy.parameters() if p.requires_grad]
        grads = [torch.zeros_like(p) for p in params]
        loss_fn = nn.CrossEntropyLoss()
        for s, a in loader:
            logits = self.policy(s)
            loss = loss_fn(logits, a)
            self.policy.zero_grad()
            loss.backward()
            for g, p in zip(grads, params):
                g += p.grad.detach().clone()
        return grads

    def _apply_gradients(self, flat: torch.Tensor) -> None:
        start = 0
        for p in self.policy.parameters():
            num = p.numel()
            g = flat[start : start + num].view_as(p)
            p.data -= self.cfg.lr * g
            start += num

    def train(self, num_agents: int) -> PolicyNet:
        envs = [SimpleEnv(self.sp_cfg.env_state_dim) for _ in range(num_agents)]
        for _ in range(self.cfg.rounds):
            enc_grads = []
            proofs: list[str] | None = [] if self.learner.require_proof else None
            for env in envs:
                states, actions = self._collect_experience(env)
                grads = self._local_gradients(states, actions)
                flat = torch.cat([g.view(-1) for g in grads])
                if self.learner.require_proof:
                    assert isinstance(proofs, list)
                    proofs.append(self.zk.generate_proof(flat))
                enc_grads.append(self.learner.encrypt(flat))
            agg = self.learner.aggregate(
                [self.learner.decrypt(g) for g in enc_grads], proofs=proofs
            )
            self._apply_gradients(agg)
        return self.policy


__all__ = [
    "FederatedRLTrainer",
    "FederatedRLTrainerConfig",
    "PolicyNet",
]
