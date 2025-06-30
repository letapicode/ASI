from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn

__all__ = ["CalibrationModel", "calibrate"]


class CalibrationModel(nn.Module):
    """Map sensor readings to calibrated outputs."""

    def __init__(self, sensor_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sensor_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, sensor_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def calibrate(model: CalibrationModel, data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
              optim: torch.optim.Optimizer, epochs: int = 1) -> None:
    model.train()
    for _ in range(epochs):
        for sample, target in data:
            optim.zero_grad()
            pred = model(sample)
            loss = nn.functional.mse_loss(pred, target)
            loss.backward()
            optim.step()
