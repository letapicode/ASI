from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple, List

import torch

from .self_play_env import SimpleEnv, rollout_env
from .robot_skill_transfer import (
    SkillTransferConfig,
    VideoPolicyDataset,
    SkillTransferModel,
    transfer_skills,
)



@dataclass
class SelfPlaySkillLoopConfig:
    """Configuration for :func:`run_loop`."""

    env_state_dim: int = 4
    img_channels: int = 3
    action_dim: int = 2
    hidden_dim: int = 128
    lr: float = 1e-4
    batch_size: int = 16
    epochs: int = 1
    steps: int = 20
    cycles: int = 3


def run_loop(
    cfg: SelfPlaySkillLoopConfig,
    policy: Callable[[torch.Tensor], torch.Tensor],
    frames: Iterable[torch.Tensor],
    actions: Iterable[int],
) -> Tuple[List[float], SkillTransferModel]:
    """Run alternating self-play and skill transfer cycles."""

    env = SimpleEnv(cfg.env_state_dim)
    dataset = VideoPolicyDataset(list(frames), list(actions))
    transfer_cfg = SkillTransferConfig(
        img_channels=cfg.img_channels,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
        lr=cfg.lr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
    )

    history: List[float] = []
    current_policy = policy
    model: SkillTransferModel | None = None
    for _ in range(cfg.cycles):
        _, rewards = rollout_env(env, current_policy, steps=cfg.steps)
        history.append(sum(rewards) / len(rewards) if rewards else 0.0)
        model = transfer_skills(transfer_cfg, dataset)

        def new_policy(obs: torch.Tensor, m: SkillTransferModel = model) -> torch.Tensor:
            with torch.no_grad():
                logits = m(obs.unsqueeze(0))
                return logits.argmax(dim=-1).squeeze(0)

        current_policy = new_policy

    assert model is not None
    return history, model


def self_play_skill_loop(
    env: SimpleEnv,
    policy: Callable[[torch.Tensor], torch.Tensor],
    skill_cfg: SkillTransferConfig,
    real_dataset: VideoPolicyDataset,
    cycles: int = 3,
    steps: int = 20,
) -> list[list[float]]:
    """Backward-compatible wrapper around :func:`run_loop`."""

    cfg = SelfPlaySkillLoopConfig(
        env_state_dim=env.state.size(0),
        img_channels=skill_cfg.img_channels,
        action_dim=skill_cfg.action_dim,
        hidden_dim=skill_cfg.hidden_dim,
        lr=skill_cfg.lr,
        batch_size=skill_cfg.batch_size,
        epochs=skill_cfg.epochs,
        steps=steps,
        cycles=cycles,
    )
    rewards, _ = run_loop(cfg, policy, real_dataset.frames, real_dataset.actions)
    return [[r] for r in rewards]


def main(argv: Sequence[str] | None = None) -> None:
    """Simple CLI for running a toy self-play loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Run self-play skill loop")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)

    cfg = SelfPlaySkillLoopConfig(cycles=args.cycles, steps=args.steps)
    policy = lambda obs: torch.zeros_like(obs)
    frames = [torch.randn(cfg.img_channels, 8, 8) for _ in range(4)]
    actions = [0 for _ in frames]
    rewards, _ = run_loop(cfg, policy, frames, actions)
    for i, r in enumerate(rewards):
        print(f"Cycle {i}: mean reward {r:.4f}")


__all__ = ["run_loop", "SelfPlaySkillLoopConfig", "self_play_skill_loop"]
