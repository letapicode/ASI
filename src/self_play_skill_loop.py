from __future__ import annotations

import logging
from typing import Callable

import torch

from .self_play_env import SimpleEnv, rollout_env
from .robot_skill_transfer import (
    SkillTransferConfig,
    VideoPolicyDataset,
    transfer_skills,
)



def self_play_skill_loop(
    env: SimpleEnv,
    policy: Callable[[torch.Tensor], torch.Tensor],
    skill_cfg: SkillTransferConfig,
    real_dataset: VideoPolicyDataset,
    cycles: int = 3,
    steps: int = 20,
) -> list[list[float]]:
    """Run alternating self-play and skill transfer cycles.

    Each cycle runs ``rollout_env`` to generate a reward trajectory then
    fine-tunes the policy using ``transfer_skills`` on ``real_dataset``.
    The reward trajectory for each cycle is logged and returned.
    """

    reward_logs: list[list[float]] = []
    for i in range(cycles):
        _, rewards = rollout_env(env, policy, steps=steps)
        logging.info("cycle %d rewards: %s", i, rewards)
        reward_logs.append(rewards)
        transfer_skills(skill_cfg, real_dataset)
    return reward_logs


__all__ = ["self_play_skill_loop"]
