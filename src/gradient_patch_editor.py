from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

@dataclass
class PatchConfig:
    lr: float = 1e-3
    steps: int = 10

class GradientPatchEditor:
    """Apply small gradient updates to fix targeted outputs."""

    def __init__(self, model: nn.Module, params: Sequence[str] | None = None, cfg: PatchConfig | None = None) -> None:
        self.model = model
        self.params = [p for n, p in model.named_parameters() if params is None or n in params]
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.params:
            p.requires_grad_(True)
        self.cfg = cfg or PatchConfig()

    def patch(self, inputs: torch.Tensor, targets: torch.Tensor, loss_fn: nn.Module) -> float:
        opt = torch.optim.SGD(self.params, lr=self.cfg.lr)
        loss_val = 0.0
        for _ in range(self.cfg.steps):
            out = self.model(inputs)
            loss = loss_fn(out, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
        return loss_val

__all__ = ["GradientPatchEditor", "PatchConfig"]
