from __future__ import annotations

import argparse
from typing import Callable

import torch

from .self_play_env import SimpleEnv, rollout_env
from .robot_skill_transfer import (
    SkillTransferConfig,
    VideoPolicyDataset,
    transfer_skills,
)



def run_loop(
    cycles: int,
    env: SimpleEnv,
    policy: Callable[[torch.Tensor], torch.Tensor],
    dataset: VideoPolicyDataset,
    cfg: SkillTransferConfig,
) -> None:
    """Alternate self-play rollouts with skill transfer fine-tuning."""
    for i in range(cycles):
        _, rewards = rollout_env(env, policy)
        avg_reward = sum(rewards) / len(rewards)
        model = transfer_skills(cfg, dataset)
        with torch.no_grad():
            frames = torch.stack([f for f, _ in dataset])
            actions = torch.tensor([a for _, a in dataset])
            preds = model(frames).argmax(dim=1)
            accuracy = (preds == actions).float().mean().item()
        print(f"Cycle {i+1}: reward {avg_reward:.3f}, skill acc {accuracy:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run self-play skill loop")
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args(argv)

    env = SimpleEnv(state_dim=4)

    def policy(obs: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(obs) * 0.1

    frames = torch.randn(8, 3, 32, 32)
    actions = torch.randint(0, 5, (8,))
    dataset = VideoPolicyDataset(frames, actions)
    cfg = SkillTransferConfig(img_channels=3, action_dim=5, epochs=1)
    run_loop(args.cycles, env, policy, dataset, cfg)


if __name__ == "__main__":
    main()
