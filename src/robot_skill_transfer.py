from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn

__all__ = ["SkillTransferModel", "transfer_skills"]


class SkillTransferModel(nn.Module):
    """Map visual demonstrations to robot actions."""

    def __init__(self, img_channels: int = 3, action_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.policy = nn.Linear(hidden, action_dim)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(imgs).squeeze(-1).squeeze(-1)
        return self.policy(h)


def transfer_skills(model: SkillTransferModel, demos: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                     optimizer: torch.optim.Optimizer, epochs: int = 1) -> None:
    """Fine-tune ``model`` on a set of (image, action) demonstrations."""
    model.train()
    for _ in range(epochs):
        for img, action in demos:
            optimizer.zero_grad()
            pred = model(img.unsqueeze(0))
            loss = nn.functional.mse_loss(pred.squeeze(0), action)
            loss.backward()
            optimizer.step()
