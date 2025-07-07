from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse)


@dataclass
class NeRFConfig:
    hidden_dim: int = 64
    lr: float = 5e-4
    epochs: int = 10
    batch_size: int = 1024
    num_samples: int = 32
    near: float = 2.0
    far: float = 6.0


class RayDataset(Dataset):
    """Collection of rays with RGB targets."""

    def __init__(self, rays: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.data = list(rays)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx]


class MultiViewDataset(RayDataset):
    """Generate rays from multi-view images and poses."""

    def __init__(
        self,
        images: Iterable[torch.Tensor],
        poses: Iterable[torch.Tensor],
        intrinsics: torch.Tensor,
    ) -> None:
        rays: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for img, pose in zip(images, poses):
            ro, rd, rgb = image_to_rays(img, pose, intrinsics)
            rays.extend(list(zip(ro, rd, rgb)))
        super().__init__(rays)
        self.height = images[0].shape[1]
        self.width = images[0].shape[2]
        self.intrinsics = intrinsics


def image_to_rays(
    image: torch.Tensor, pose: torch.Tensor, intrinsics: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c, h, w = image.shape
    i, j = torch.meshgrid(
        torch.arange(w, dtype=torch.float32),
        torch.arange(h, dtype=torch.float32),
        indexing="ij",
    )
    dirs = torch.stack([
        (i - intrinsics[0, 2]) / intrinsics[0, 0],
        (j - intrinsics[1, 2]) / intrinsics[1, 1],
        torch.ones_like(i),
    ], dim=-1)
    dirs = dirs.reshape(-1, 3)
    rays_d = (pose[:3, :3] @ dirs.T).T
    rays_o = pose[:3, 3].expand_as(rays_d)
    rgb = image.permute(1, 2, 0).reshape(-1, c)
    return rays_o, rays_d, rgb


class TinyNeRF(nn.Module):
    """Minimal NeRF network."""

    def __init__(self, cfg: NeRFConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(3, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def render(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        t_vals = torch.linspace(
            self.cfg.near,
            self.cfg.far,
            self.cfg.num_samples,
            device=rays_o.device,
        )
        pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[None, :, None]
        raw = self(pts.reshape(-1, 3)).view(rays_o.shape[0], self.cfg.num_samples, 4)
        rgb = torch.sigmoid(raw[..., :3])
        sigma = torch.relu(raw[..., 3])
        deltas = t_vals[1:] - t_vals[:-1]
        deltas = torch.cat([deltas, torch.tensor([1e10], device=deltas.device)])
        alpha = 1.0 - torch.exp(-sigma * deltas)
        trans = torch.cumprod(
            torch.cat([
                torch.ones((rays_o.shape[0], 1), device=rays_o.device),
                1 - alpha + 1e-10,
            ], dim=1),
            dim=1,
        )[:, :-1]
        weights = alpha * trans
        color = (weights[..., None] * rgb).sum(dim=1)
        return color


def train_nerf(cfg: NeRFConfig, dataset: Dataset) -> TinyNeRF:
    model = TinyNeRF(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(cfg.epochs):
        for ro, rd, target in loader:
            pred = model.render(ro, rd)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def render_views(
    model: TinyNeRF,
    poses: Iterable[torch.Tensor],
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
) -> List[torch.Tensor]:
    h, w = image_size
    dummy = torch.zeros(3, h, w)
    frames = []
    for pose in poses:
        ro, rd, _ = image_to_rays(dummy, pose, intrinsics)
        img = model.render(ro, rd).reshape(h, w, 3).permute(2, 0, 1)
        frames.append(img)
    return frames


__all__ = [
    "NeRFConfig",
    "RayDataset",
    "MultiViewDataset",
    "TinyNeRF",
    "train_nerf",
    "render_views",
    "psnr",
]
