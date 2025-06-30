from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn

__all__ = ["collect_trajectories", "ModelBasedAgent", "train_model", "evaluate"]


def collect_trajectories(env, policy, num_steps: int) -> list[Tuple[np.ndarray, np.ndarray, float]]:
    """Collect (state, action, reward) tuples from ``env`` using ``policy``."""
    traj = []
    state, _ = env.reset()
    for _ in range(num_steps):
        action = policy(state)
        next_state, reward, done, truncated, _ = env.step(action)
        traj.append((state, action, reward))
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    return traj


@dataclass
class ModelBasedAgent:
    model: nn.Module
    optimizer: torch.optim.Optimizer

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.model(x)
            return torch.argmax(logits, dim=-1).cpu().numpy()[0]


def train_model(model: nn.Module, data: Iterable[Tuple[np.ndarray, np.ndarray, float]],
                optimizer: torch.optim.Optimizer, epochs: int = 1) -> None:
    for _ in range(epochs):
        for state, action, reward in data:
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            target = torch.as_tensor(action, dtype=torch.long).unsqueeze(0)
            optimizer.zero_grad()
            out = model(state)
            loss = nn.functional.cross_entropy(out, target)
            loss.backward()
            optimizer.step()


def evaluate(env, agent: ModelBasedAgent, episodes: int = 5) -> float:
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(obs)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            if truncated:
                break
        total += ep_reward
    return total / episodes
