import random
from typing import Iterable, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """Simple feed-forward world model predicting next state and reward."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim + 1),
        )
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        out = self.net(x)
        next_obs = out[..., : self.obs_dim]
        reward = out[..., self.obs_dim :]
        return next_obs, reward.squeeze(-1)


def train_world_model(
    data: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    epochs: int = 1,
) -> None:
    """Train ``model`` to predict next observations and rewards."""
    model.train()
    for _ in range(epochs):
        for obs, action, next_obs, reward in data:
            pred_obs, pred_rew = model(obs, action)
            loss = F.mse_loss(pred_obs, next_obs) + F.mse_loss(pred_rew, reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def rollout(
    model: WorldModel,
    policy: "Policy",
    start_state: torch.Tensor,
    horizon: int,
) -> Tuple[List[torch.Tensor], List[float]]:
    """Simulate ``horizon`` steps using ``model`` and ``policy``."""
    state = start_state
    states = [state]
    rewards: List[float] = []
    for _ in range(horizon):
        action = policy.act(state)
        with torch.no_grad():
            state, reward = model(state.unsqueeze(0), action.unsqueeze(0))
            state = state.squeeze(0)
            reward = reward.item()
        states.append(state)
        rewards.append(reward)
    return states, rewards


class Policy:
    """Base policy interface."""

    def act(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def act(self, state: torch.Tensor) -> torch.Tensor:
        return torch.rand(self.action_dim, device=state.device)


def collect_dataset(
    env,
    policy: Policy,
    episodes: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate an offline dataset from ``env`` using ``policy``."""
    dataset = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy.act(torch.tensor(obs, dtype=torch.float32))
            next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
            dataset.append(
                (
                    torch.tensor(obs, dtype=torch.float32),
                    action.float(),
                    torch.tensor(next_obs, dtype=torch.float32),
                    torch.tensor(reward, dtype=torch.float32),
                )
            )
            obs = next_obs
            done = terminated or truncated
    return dataset

