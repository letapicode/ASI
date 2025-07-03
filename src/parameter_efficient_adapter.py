from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from .low_rank_adapter import apply_low_rank_adaptation


@dataclass
class PEFTConfig:
    target_modules: Sequence[str]
    r: int = 4
    alpha: float = 1.0
    lr: float = 1e-3
    epochs: int = 1


class ParameterEfficientAdapter:
    """Fine-tune only low-rank adapters for new tasks."""

    def __init__(self, model: nn.Module, cfg: PEFTConfig) -> None:
        self.model = apply_low_rank_adaptation(
            model, cfg.target_modules, r=cfg.r, alpha=cfg.alpha
        )
        # collect adapter parameters
        self.params: list[nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if "A" in name or "B" in name:
                p.requires_grad = True
                self.params.append(p)
            else:
                p.requires_grad = False
        self.cfg = cfg
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr)

    def fit(self, dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]], loss_fn: nn.Module) -> None:
        self.model.train()
        for _ in range(self.cfg.epochs):
            for x, y in dataloader:
                out = self.model(x)
                loss = loss_fn(out, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


__all__ = ["PEFTConfig", "ParameterEfficientAdapter"]
