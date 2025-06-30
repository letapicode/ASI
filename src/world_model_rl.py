import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, Tuple, List
import gym


class WorldModel(nn.Module):
    """Generative world model for model-based RL."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.obs_embed = nn.Linear(obs_dim, hidden)
        self.action_embed = nn.Embedding(action_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.out_state = nn.Linear(hidden, obs_dim)
        self.out_reward = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.obs_embed(obs) + self.action_embed(action)
        h, _ = self.rnn(x)
        next_obs = self.out_state(h)
        reward = self.out_reward(h).squeeze(-1)
        return next_obs, reward


def train_world_model(
    model: WorldModel,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int = 1,
    lr: float = 1e-3,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for obs, action, next_obs, reward in dataset:
            pred_state, pred_reward = model(obs, action)
            loss = F.mse_loss(pred_state, next_obs) + F.mse_loss(pred_reward, reward)
            opt.zero_grad()
            loss.backward()
            opt.step()


def rollout_model(
    model: WorldModel,
    start_obs: torch.Tensor,
    policy: nn.Module,
    steps: int = 20,
) -> Tuple[List[torch.Tensor], List[float]]:
    model.eval()
    obs = start_obs
    states: List[torch.Tensor] = [obs]
    rewards: List[float] = []
    for _ in range(steps):
        with torch.no_grad():
            act_logits = policy(obs)
            action = act_logits.argmax(-1)
            next_obs, reward = model(obs.unsqueeze(0), action.unsqueeze(0))
            next_obs = next_obs.squeeze(0)
            reward = reward.item()
        states.append(next_obs)
        rewards.append(reward)
        obs = next_obs
    return states, rewards


def model_based_rl(
    env: gym.Env,
    model: WorldModel,
    policy: nn.Module,
    episodes: int = 10,
    rollout_len: int = 20,
    gamma: float = 0.99,
) -> None:
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for _ in range(episodes):
        obs = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        while not done:
            act_logits = policy(obs)
            action = int(act_logits.argmax().item())
            next_obs, reward, done, _ = env.step(action)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32)
            loss = -reward
            # imaginary rollout for value target
            sims, rews = rollout_model(model, next_obs_t, policy, steps=rollout_len)
            ret = sum(r * (gamma ** i) for i, r in enumerate(rews))
            loss += -ret
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            obs = next_obs_t
