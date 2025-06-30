from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class SkillTransferConfig:
    img_channels: int
    action_dim: int
    hidden_dim: int = 128
    lr: float = 1e-4
    epochs: int = 5
    batch_size: int = 16


class VideoPolicyDataset(Dataset):
    """Video frames paired with robot actions."""

    def __init__(self, frames: Iterable[Any], actions: Iterable[int]):
        self.frames = list(frames)
        self.actions = list(actions)
        if len(self.frames) != len(self.actions):
            raise ValueError("frames and actions must have same length")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        return self.frames[idx], self.actions[idx]


class SkillTransferModel(nn.Module):
    """Map video frames to discrete robot actions."""

    def __init__(self, cfg: SkillTransferConfig) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.img_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, cfg.hidden_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return self.fc(h)


def transfer_skills(cfg: SkillTransferConfig, dataset: VideoPolicyDataset) -> SkillTransferModel:
    """Fine-tune policy on real robot examples."""
    model = SkillTransferModel(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg.epochs):
        for frames, actions in loader:
            frames = frames.to(device)
            actions = actions.to(device)
            logits = model(frames)
            loss = loss_fn(logits, actions)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


__all__ = ["SkillTransferConfig", "VideoPolicyDataset", "SkillTransferModel", "transfer_skills"]
