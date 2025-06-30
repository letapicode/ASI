from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


@dataclass
class CalibrationConfig:
    sensor_dim: int
    actuator_dim: int
    hidden_dim: int = 64
    lr: float = 1e-3
    epochs: int = 20


class CalibrationDataset(torch.utils.data.Dataset):
    """Pairs of (sim_sensor, real_sensor, real_actuator)."""

    def __init__(self, entries: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.data = list(entries)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class CalibrationModel(torch.nn.Module):
    """Translate sim observations to real actuator parameters."""

    def __init__(self, cfg: CalibrationConfig) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(cfg.sensor_dim * 2, cfg.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.hidden_dim, cfg.actuator_dim),
        )

    def forward(self, sim_s: torch.Tensor, real_s: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([sim_s, real_s], dim=-1))


def calibrate(cfg: CalibrationConfig, dataset: CalibrationDataset) -> CalibrationModel:
    model = CalibrationModel(cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.MSELoss()
    device = next(model.parameters()).device
    model.train()
    for _ in range(cfg.epochs):
        for sim_s, real_s, target_act in loader:
            sim_s = sim_s.to(device)
            real_s = real_s.to(device)
            target_act = target_act.to(device)
            pred = model(sim_s, real_s)
            loss = loss_fn(pred, target_act)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


__all__ = ["CalibrationConfig", "CalibrationDataset", "CalibrationModel", "calibrate"]
