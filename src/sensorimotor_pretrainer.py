from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .multimodal_world_model import MultiModalWorldModel


@dataclass
class SensorimotorPretrainConfig:
    """Hyperparameters for sensorimotor pretraining."""

    epochs: int = 1
    batch_size: int = 8


class SensorimotorLogDataset(Dataset):
    """Simple container for unlabeled sensorimotor transition logs."""

    def __init__(self, entries: Iterable[Tuple[Any, Any, Any, Any, Any]], tokenizer) -> None:
        self.data = list(entries)
        self.tokenizer = tokenizer

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int):  # type: ignore[override]
        t, img, a, nt, nimg = self.data[idx]
        return (
            torch.tensor(self.tokenizer(t), dtype=torch.long),
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(self.tokenizer(nt), dtype=torch.long),
            torch.tensor(nimg, dtype=torch.float32),
        )


def pretrain_sensorimotor(
    model: MultiModalWorldModel,
    dataset: Dataset,
    cfg: SensorimotorPretrainConfig,
) -> MultiModalWorldModel:
    """Train ``model`` to predict next observations in an unsupervised manner."""

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=model.cfg.lr)
    device = next(model.parameters()).device
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(cfg.epochs):
        for t, img, a, nt, nimg in loader:
            t = t.to(device)
            img = img.to(device)
            a = a.to(device)
            nt = nt.to(device)
            nimg = nimg.to(device)
            state = model.encode_obs(t, img)
            pred_state, _ = model.predict_dynamics(state, a)
            target = model.encode_obs(nt, nimg)
            loss = loss_fn(pred_state, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


__all__ = [
    "SensorimotorPretrainConfig",
    "SensorimotorLogDataset",
    "pretrain_sensorimotor",
]
