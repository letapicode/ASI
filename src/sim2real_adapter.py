from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


@dataclass
class Sim2RealConfig:
    state_dim: int
    lr: float = 1e-3
    epochs: int = 5


class Sim2RealAdapter(torch.nn.Module):
    """Learn a simple linear map from simulated to real states."""

    def __init__(self, cfg: Sim2RealConfig) -> None:
        super().__init__()
        self.net = torch.nn.Linear(cfg.state_dim, cfg.state_dim)
        self.cfg = cfg

    def forward(self, sim_state: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(sim_state)

    def fit(self, logs: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        data = list(logs)
        if not data:
            return
        loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        loss_fn = torch.nn.MSELoss()
        self.train()
        for _ in range(self.cfg.epochs):
            for sim_s, real_s in loader:
                pred = self(sim_s)
                loss = loss_fn(pred, real_s)
                opt.zero_grad()
                loss.backward()
                opt.step()


__all__ = ["Sim2RealConfig", "Sim2RealAdapter"]
