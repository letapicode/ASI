from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any

try:
    from .self_play_env import SimpleEnv
    from .self_play_skill_loop import SelfPlaySkillLoopConfig, run_loop as _run_loop
    from .robot_skill_transfer import SkillTransferModel
    from . import self_play_skill_loop
    from .differential_privacy_optimizer import (
        DifferentialPrivacyOptimizer,
        DifferentialPrivacyConfig,
    )
    from .privacy_budget_manager import PrivacyBudgetManager
except Exception:  # pragma: no cover - fallback for tests
    import importlib.util
    import sys
    from pathlib import Path

    base = Path(__file__).parent

    def _load(name: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, base / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    SimpleEnv = _load("self_play_env").SimpleEnv  # type: ignore
    SkillTransferModel = _load("robot_skill_transfer").SkillTransferModel  # type: ignore
    skl = _load("self_play_skill_loop")
    SelfPlaySkillLoopConfig = skl.SelfPlaySkillLoopConfig  # type: ignore
    _run_loop = skl.run_loop  # type: ignore
    self_play_skill_loop = skl
    dpo = _load("differential_privacy_optimizer")
    DifferentialPrivacyOptimizer = dpo.DifferentialPrivacyOptimizer  # type: ignore
    DifferentialPrivacyConfig = dpo.DifferentialPrivacyConfig
    pbm_mod = _load("privacy_budget_manager")
    PrivacyBudgetManager = pbm_mod.PrivacyBudgetManager

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from .compute_budget_tracker import ComputeBudgetTracker
try:
    from .budget_aware_scheduler import BudgetAwareScheduler
except Exception:  # pragma: no cover - for tests
    BudgetAwareScheduler = None


@dataclass
class RLBridgeConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 10


class TransitionDataset(Dataset):
    """Logged ``(state, action, next_state, reward)`` tuples."""

    def __init__(self, transitions: Iterable[tuple[torch.Tensor, int, torch.Tensor, float]]):
        self.data = list(transitions)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class TrajectoryDataset(TransitionDataset):
    """Alias for :class:`TransitionDataset` used by self-play helpers."""

    pass


class WorldModel(nn.Module):
    """Simple predictive model for RL."""

    def __init__(self, cfg: RLBridgeConfig) -> None:
        super().__init__()
        self.state_fc = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.action_emb = nn.Embedding(cfg.action_dim, cfg.hidden_dim)
        self.out_state = nn.Linear(cfg.hidden_dim, cfg.state_dim)
        self.out_reward = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.state_fc(state) + self.action_emb(action))
        next_state = self.out_state(h)
        reward = self.out_reward(h).squeeze(-1)
        return next_state, reward


def train_world_model(
    cfg: RLBridgeConfig,
    dataset: Dataset,
    dp_cfg: DifferentialPrivacyConfig | None = None,
    pbm: "PrivacyBudgetManager | None" = None,
    run_id: str = "default",
    budget: ComputeBudgetTracker | None = None,
    synth_3d: Iterable[tuple[str, np.ndarray]] | None = None,
) -> WorldModel:
    model = WorldModel(cfg)
    if synth_3d is not None:
        extra: list[tuple[torch.Tensor, int, torch.Tensor, float]] = []
        for _txt, vol in synth_3d:
            t = torch.tensor(vol, dtype=torch.float32).view(-1)
            if t.numel() < cfg.state_dim:
                pad = torch.zeros(cfg.state_dim - t.numel())
                t = torch.cat([t, pad])
            elif t.numel() > cfg.state_dim:
                t = t[: cfg.state_dim]
            extra.append((t, 0, t.clone(), 0.0))
        dataset = ConcatDataset([dataset, TransitionDataset(extra)])
    scheduler = (
        BudgetAwareScheduler(budget, run_id)
        if budget is not None and BudgetAwareScheduler is not None
        else None
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    if dp_cfg is None:
        opt: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        opt = DifferentialPrivacyOptimizer(model.parameters(), dp_cfg)
    loss_fn = nn.MSELoss()
    device = next(model.parameters()).device
    model.train()
    for _ in range(cfg.epochs):
        if scheduler is not None:
            scheduler.schedule_step(cfg)
            loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        for state, action, next_state, reward in loader:
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            reward = reward.to(device)
            pred_state, pred_reward = model(state, action)
            loss = loss_fn(pred_state, next_state) + loss_fn(pred_reward, reward)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if budget is not None and budget.remaining(run_id) <= 0.0:
                break
        if budget is not None and budget.remaining(run_id) <= 0.0:
            break
    if pbm is not None and dp_cfg is not None:
        eps = dp_cfg.noise_std * len(dataset) / cfg.batch_size
        pbm.consume(run_id, eps, 0.0)
    return model


def rollout_policy(model: WorldModel, policy: Callable[[torch.Tensor], torch.Tensor], init_state: torch.Tensor, steps: int = 50) -> tuple[list[torch.Tensor], list[float]]:
    device = next(model.parameters()).device
    state = init_state.to(device)
    states = []
    rewards = []
    with torch.no_grad():
        for _ in range(steps):
            action = policy(state)
            next_state, reward = model(state, action)
            states.append(next_state.cpu())
            rewards.append(float(reward.item()))
            state = next_state
    return states, rewards


def train_with_self_play(
    rl_cfg: RLBridgeConfig,
    sp_cfg: "SelfPlaySkillLoopConfig",
    policy: Callable[[torch.Tensor], torch.Tensor],
    frames: Iterable[torch.Tensor],
    actions: Iterable[int],
    dp_cfg: DifferentialPrivacyConfig | None = None,
    sampler_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[WorldModel, SkillTransferModel]:
    """Run self-play to gather transitions and fit a world model."""

    transitions: list[tuple[torch.Tensor, int, torch.Tensor, float]] = []

    def record_rollout(
        env: "SimpleEnv",
        pol: Callable[[torch.Tensor], torch.Tensor],
        steps: int = sp_cfg.steps,
        return_actions: bool = False,
    ):
        obs = env.reset()
        observations: list[torch.Tensor] = []
        rewards: list[float] = []
        acts: list[int] = []
        for _ in range(steps):
            raw_a = pol(obs)
            a = sampler_fn(raw_a) if sampler_fn is not None else raw_a
            step = env.step(a)
            transitions.append(
                (
                    obs.clone(),
                    int(a if not isinstance(a, torch.Tensor) else a.item()),
                    step.observation.clone(),
                    step.reward,
                )
            )
            observations.append(step.observation)
            rewards.append(step.reward)
            if return_actions:
                acts.append(int(a if not isinstance(a, torch.Tensor) else a.item()))
            obs = step.observation
            if step.done:
                break
        if return_actions:
            return observations, rewards, acts
        return observations, rewards

    orig = self_play_skill_loop.rollout_env
    self_play_skill_loop.rollout_env = record_rollout  # type: ignore
    try:
        _, skill_model = _run_loop(sp_cfg, policy, frames, actions)
    finally:
        self_play_skill_loop.rollout_env = orig  # type: ignore

    dataset = TrajectoryDataset(transitions)
    wm = train_world_model(rl_cfg, dataset, dp_cfg)
    return wm, skill_model


__all__ = [
    "RLBridgeConfig",
    "TransitionDataset",
    "TrajectoryDataset",
    "WorldModel",
    "train_world_model",
    "train_with_self_play",
    "rollout_policy",
]
