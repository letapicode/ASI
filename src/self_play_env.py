from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable, Tuple, List

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


class PrioritizedReplayBuffer:
    """Replay buffer that samples transitions with probability proportional
    to positive reward."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: List[Tuple[torch.Tensor, int, float]] = []
        self.pos = 0

    def add(self, obs: torch.Tensor, action: int, reward: float) -> None:
        item = (obs.detach().clone(), int(action), float(reward))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[torch.Tensor], List[int]]:
        if not self.data:
            raise ValueError("buffer empty")
        weights = torch.tensor([max(r, 1e-6) for (_, _, r) in self.data], dtype=torch.float)
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, batch_size, replacement=True).tolist()
        frames = [self.data[i][0] for i in idx]
        actions = [self.data[i][1] for i in idx]
        return frames, actions


def rollout_env(
    env: SimpleEnv,
    policy: Callable[[torch.Tensor], torch.Tensor],
    steps: int = 20,
    return_actions: bool = False,
) -> Tuple[list[torch.Tensor], list[float]] | Tuple[list[torch.Tensor], list[float], list[int]]:
    """Run environment with policy returning actions."""
    obs = env.reset()
    observations: list[torch.Tensor] = []
    rewards: list[float] = []
    actions: list[int] = []
    for _ in range(steps):
        action_t = policy(obs)
        step = env.step(action_t)
        observations.append(step.observation)
        rewards.append(step.reward)
        if return_actions:
            if isinstance(action_t, torch.Tensor):
                actions.append(int(action_t.item()))
            else:
                actions.append(int(action_t))
        obs = step.observation
        if step.done:
            break
    if return_actions:
        return observations, rewards, actions
    return observations, rewards


__all__ = ["EnvStep", "SimpleEnv", "PrioritizedReplayBuffer", "rollout_env"]
