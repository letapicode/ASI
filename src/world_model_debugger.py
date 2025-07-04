from __future__ import annotations

from typing import Iterable
import torch
from torch import nn

from .gradient_patch_editor import GradientPatchEditor, PatchConfig


class WorldModelDebugger:
    """Monitor rollout errors and apply gradient patches when needed."""

    def __init__(self, model: nn.Module, threshold: float = 1.0, cfg: PatchConfig | None = None) -> None:
        self.model = model
        self.threshold = threshold
        self.editor = GradientPatchEditor(model, cfg=cfg)
        self.loss_fn = nn.MSELoss()

    def check(self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor) -> float:
        pred, _ = self.model(states, actions)
        loss = self.loss_fn(pred, targets)
        if loss.item() > self.threshold:
            opt = torch.optim.SGD(self.model.parameters(), lr=self.editor.cfg.lr)
            for _ in range(self.editor.cfg.steps):
                p, _ = self.model(states, actions)
                l = self.loss_fn(p, targets)
                opt.zero_grad()
                l.backward()
                opt.step()
            loss = l
        return float(loss.item())


__all__ = ["WorldModelDebugger"]
