from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple, List

import torch

try:
    from .deliberative_alignment import DeliberativeAligner
except Exception:  # pragma: no cover - fallback for tests
    import importlib.util
    import sys
    from pathlib import Path

    base = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "deliberative_alignment", base / "deliberative_alignment.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["deliberative_alignment"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    DeliberativeAligner = mod.DeliberativeAligner

try:
    from .self_play_env import SimpleEnv, rollout_env, PrioritizedReplayBuffer
    from .robot_skill_transfer import (
        SkillTransferConfig,
        VideoPolicyDataset,
        SkillTransferModel,
        transfer_skills,
    )
except Exception:  # pragma: no cover - fallback for tests
    from self_play_env import SimpleEnv, rollout_env, PrioritizedReplayBuffer  # type: ignore
    from robot_skill_transfer import (
        SkillTransferConfig,  # type: ignore
        VideoPolicyDataset,
        SkillTransferModel,
        transfer_skills,
    )


class SafetyPolicyMonitor:
    """Check alignment each cycle."""

    def __init__(self, policy: str) -> None:
        self.aligner = DeliberativeAligner(policy)
        self.violations: list[str] = []

    def check(self, transcript: str) -> None:
        if not self.aligner.check(transcript.split("\n")):
            self.violations.append(transcript)


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
    monitor: SafetyPolicyMonitor | None = None,
) -> Tuple[List[float], SkillTransferModel]:
    """Run alternating self-play and skill transfer cycles."""

    env = SimpleEnv(cfg.env_state_dim)
    buffer = PrioritizedReplayBuffer(capacity=cfg.steps * cfg.cycles)
    for f, a in zip(frames, actions):
        buffer.add(f, a, 1.0)
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
        obs_list, rewards, acts = rollout_env(
            env, current_policy, steps=cfg.steps, return_actions=True
        )
        for o, a, r in zip(obs_list, acts, rewards):
            buffer.add(o, a, r)
        if monitor is not None:
            transcript = "\n".join(f"{o.tolist()}:{a}" for o, a in zip(obs_list, acts))
            monitor.check(transcript)
        history.append(sum(rewards) / len(rewards) if rewards else 0.0)
        sample_frames, sample_actions = buffer.sample_by_priority(cfg.batch_size)
        dataset = VideoPolicyDataset(sample_frames, sample_actions)
        model = transfer_skills(transfer_cfg, dataset)

        def new_policy(
            obs: torch.Tensor, m: SkillTransferModel = model
        ) -> torch.Tensor:
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
    monitor: SafetyPolicyMonitor | None = None,
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
    rewards, _ = run_loop(
        cfg, policy, real_dataset.frames, real_dataset.actions, monitor
    )
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


__all__ = [
    "run_loop",
    "SelfPlaySkillLoopConfig",
    "self_play_skill_loop",
    "SafetyPolicyMonitor",
]
