from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

from .self_play_env import SimpleEnv, rollout_env
from .robot_skill_transfer import SkillTransferConfig, VideoPolicyDataset, SkillTransferModel, transfer_skills


@dataclass
class SelfPlaySkillConfig:
    """Configuration for the integrated loop."""

    cycles: int = 3
    rollout_steps: int = 20


def self_play_skill_loop(
    env: SimpleEnv,
    policy: Callable[[torch.Tensor], torch.Tensor],
    transfer_cfg: SkillTransferConfig,
    real_dataset: VideoPolicyDataset,
    cfg: SelfPlaySkillConfig | None = None,
) -> Tuple[SkillTransferModel, List[List[float]]]:
    """Alternate self-play rollouts with skill transfer training.

    Returns the final trained policy and reward history.
    """

    if cfg is None:
        cfg = SelfPlaySkillConfig()
    rewards_history: List[List[float]] = []
    current_policy = policy
    model: SkillTransferModel | None = None

    for _ in range(cfg.cycles):
        _, rewards = rollout_env(env, current_policy, steps=cfg.rollout_steps)
        rewards_history.append(rewards)
        model = transfer_skills(transfer_cfg, real_dataset)
        current_policy = model

    return model if model is not None else policy, rewards_history


__all__ = ["SelfPlaySkillConfig", "self_play_skill_loop"]
