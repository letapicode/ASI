from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class TinyNeRFCfg:
    hidden_dim: int = 32
    lr: float = 1e-3
    epochs: int = 10


class TinyNeRF(nn.Module):
    """Minimal NeRF-style MLP for small scenes."""

    def __init__(self, cfg: TinyNeRFCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(6, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rays_o, rays_d], dim=-1)
        return self.mlp(x)

    def render(self, pose: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        ii, jj = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=pose.device),
            torch.arange(w, dtype=torch.float32, device=pose.device),
            indexing="ij",
        )
        dirs = torch.stack(
            [(jj - w / 2) / w, (ii - h / 2) / h, torch.ones_like(ii)], dim=-1
        ).view(-1, 3)
        rays_o = pose[:3, 3].expand(dirs.shape[0], 3)
        cols = self.forward(rays_o, dirs)
        return cols.t().view(3, h, w)


class MultiViewDataset(Dataset):
    """List of ``(pose, image)`` pairs used for NeRF training."""

    def __init__(self, views: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.data = list(views)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.data[idx]


class RayDataset(Dataset):
    """Flatten multi-view images into per-ray samples."""

    def __init__(self, views: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        rays_o = []
        rays_d = []
        cols = []
        for pose, img in views:
            h, w = img.shape[1], img.shape[2]
            ii, jj = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            )
            dirs = torch.stack(
                [(jj - w / 2) / w, (ii - h / 2) / h, torch.ones_like(ii)], dim=-1
            ).view(-1, 3)
            rays_o.append(pose[:3, 3].expand(dirs.shape[0], 3))
            rays_d.append(dirs)
            cols.append(img.permute(1, 2, 0).reshape(-1, 3))
        self.rays_o = torch.cat(rays_o, dim=0)
        self.rays_d = torch.cat(rays_d, dim=0)
        self.colors = torch.cat(cols, dim=0)

    def __len__(self) -> int:  # type: ignore[override]
        return self.rays_o.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.rays_o[idx], self.rays_d[idx], self.colors[idx]


def train_nerf(model: TinyNeRF, dataset: Dataset, batch_size: int = 32) -> TinyNeRF:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=model.cfg.lr)
    loss_fn = nn.MSELoss()
    for _ in range(model.cfg.epochs):
        for ro, rd, col in loader:
            pred = model(ro, rd)
            loss = loss_fn(pred, col)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


__all__ = [
    "TinyNeRFCfg",
    "TinyNeRF",
    "MultiViewDataset",
    "RayDataset",
    "train_nerf",
]
