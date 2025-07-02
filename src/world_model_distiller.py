from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .multimodal_world_model import MultiModalWorldModel


@dataclass
class DistillConfig:
    epochs: int = 1
    batch_size: int = 8
    alpha: float = 0.5  # weight on teacher predictions


def distill_world_model(
    teacher: MultiModalWorldModel,
    student: MultiModalWorldModel,
    dataset: Dataset,
    cfg: DistillConfig,
) -> MultiModalWorldModel:
    """Train ``student`` to match ``teacher`` predictions."""

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(student.parameters(), lr=student.cfg.lr)
    device = next(student.parameters()).device
    teacher.eval()
    loss_fn = nn.MSELoss()
    for _ in range(cfg.epochs):
        for t, img, a, nt, nimg, r in loader:
            t = t.to(device)
            img = img.to(device)
            a = a.to(device)
            nt = nt.to(device)
            nimg = nimg.to(device)
            r = r.to(device)
            with torch.no_grad():
                t_state = teacher.encode_obs(t, img)
                t_pred, t_reward = teacher.predict_dynamics(t_state, a)
            s_state = student.encode_obs(t, img)
            s_pred, s_reward = student.predict_dynamics(s_state, a)
            gt_state = student.encode_obs(nt, nimg)
            loss = (
                cfg.alpha
                * (loss_fn(s_pred, t_pred) + loss_fn(s_reward, t_reward))
                + (1 - cfg.alpha)
                * (loss_fn(s_pred, gt_state) + loss_fn(s_reward, r))
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return student


__all__ = ["DistillConfig", "distill_world_model"]
