import torch
from torch import nn
from torchdiffeq import odeint
from dataclasses import dataclass
from typing import Callable, Iterable

@dataclass
class ODEWorldModelConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    dt: float = 1.0

class _Dynamics(nn.Module):
    def __init__(self, cfg: ODEWorldModelConfig) -> None:
        super().__init__()
        self.state_fc = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.action_emb = nn.Embedding(cfg.action_dim, cfg.hidden_dim)
        self.out = nn.Linear(cfg.hidden_dim, cfg.state_dim)
        self._action: torch.Tensor | None = None

    def set_action(self, action: torch.Tensor) -> None:
        self._action = self.action_emb(action)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        assert self._action is not None
        h = torch.tanh(self.state_fc(state) + self._action)
        return self.out(h)

class ODEWorldModel(nn.Module):
    """Continuous-time world model using an ODE solver."""

    def __init__(self, cfg: ODEWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.func = _Dynamics(cfg)
        self.reward_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.func.set_action(action)
        t = torch.tensor([0.0, self.cfg.dt], device=state.device)
        next_state = odeint(self.func, state, t)[-1]
        h = torch.tanh(self.func.state_fc(next_state) + self.func.action_emb(action))
        reward = self.reward_head(h).squeeze(-1)
        return next_state, reward


def train_ode_world_model(cfg: ODEWorldModelConfig, dataset: Iterable[tuple[torch.Tensor, int, torch.Tensor, float]]) -> ODEWorldModel:
    model = ODEWorldModel(cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(cfg.epochs):
        for s, a, ns, r in loader:
            pred_s, pred_r = model(s, a)
            loss = loss_fn(pred_s, ns) + loss_fn(pred_r, r)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def rollout_policy(model: ODEWorldModel, policy: Callable[[torch.Tensor], torch.Tensor], init_state: torch.Tensor, steps: int = 50) -> tuple[list[torch.Tensor], list[float]]:
    device = next(model.parameters()).device
    state = init_state.to(device)
    states: list[torch.Tensor] = []
    rewards: list[float] = []
    with torch.no_grad():
        for _ in range(steps):
            action = policy(state)
            state, reward = model(state, action)
            states.append(state.cpu())
            rewards.append(float(reward.item()))
    return states, rewards

__all__ = [
    "ODEWorldModelConfig",
    "ODEWorldModel",
    "train_ode_world_model",
    "rollout_policy",
]
