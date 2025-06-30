from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable, Tuple

import torch


@dataclass
class EnvStep:
    observation: torch.Tensor
    reward: float
    done: bool


class SimpleEnv:
    """Toy environment with numeric state."""

    def __init__(self, state_dim: int, device: torch.device | None = None) -> None:
        self.state = torch.zeros(state_dim, device=device)
        self.device = device or torch.device("cpu")

    def reset(self) -> torch.Tensor:
        self.state.zero_()
        return self.state.clone()

    def step(self, action: torch.Tensor) -> EnvStep:
        self.state += action.to(self.device)
        reward = -self.state.norm().item()
        done = reward > -0.1
        return EnvStep(self.state.clone(), reward, done)


def rollout_env(env: SimpleEnv, policy: Callable[[torch.Tensor], torch.Tensor], steps: int = 20) -> Tuple[list[torch.Tensor], list[float]]:
    """Run environment with policy returning actions."""
    obs = env.reset()
    observations = []
    rewards = []
    for _ in range(steps):
        action = policy(obs)
        step = env.step(action)
        observations.append(step.observation)
        rewards.append(step.reward)
        obs = step.observation
        if step.done:
            break
    return observations, rewards


__all__ = ["EnvStep", "SimpleEnv", "rollout_env"]
