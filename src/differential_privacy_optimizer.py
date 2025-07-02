from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class DifferentialPrivacyConfig:
    lr: float = 1e-3
    clip_norm: float = 1.0
    noise_std: float = 0.01


class DifferentialPrivacyOptimizer:
    """Wrap an optimizer with gradient clipping and noise."""

    def __init__(self, params: Iterable[torch.nn.Parameter], cfg: DifferentialPrivacyConfig) -> None:
        self.cfg = cfg
        self.base_opt = torch.optim.Adam(params, lr=cfg.lr)

    def zero_grad(self) -> None:
        self.base_opt.zero_grad()

    def step(self, closure=None) -> None:
        if closure is not None:
            closure()
        torch.nn.utils.clip_grad_norm_(self.base_opt.param_groups[0]["params"], self.cfg.clip_norm)
        for group in self.base_opt.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.grad) * self.cfg.noise_std
                p.grad.add_(noise)
        self.base_opt.step()


__all__ = ["DifferentialPrivacyConfig", "DifferentialPrivacyOptimizer"]
