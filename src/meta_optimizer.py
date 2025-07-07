from __future__ import annotations

import copy
from typing import Callable, Iterable, Sequence

import torch
from torch import nn


class MetaOptimizer:
    """Tiny first-order MAML optimizer."""

    def __init__(
        self,
        train_fn: Callable[[nn.Module, Iterable], torch.Tensor],
        *,
        meta_lr: float = 1e-2,
        adapt_lr: float = 1e-2,
        adapt_steps: int = 1,
    ) -> None:
        self.train_fn = train_fn
        self.meta_lr = float(meta_lr)
        self.adapt_lr = float(adapt_lr)
        self.adapt_steps = int(adapt_steps)

    def adapt(self, model: nn.Module, task_data: Iterable) -> nn.Module:
        """Return a clone adapted to ``task_data`` for ``adapt_steps``."""
        clone = copy.deepcopy(model)
        opt = torch.optim.SGD(clone.parameters(), lr=self.adapt_lr)
        for _ in range(self.adapt_steps):
            opt.zero_grad()
            loss = self.train_fn(clone, task_data)
            loss.backward()
            opt.step()
        return clone

    def meta_step(self, model: nn.Module, tasks: Sequence[Iterable]) -> float:
        """Perform one meta-update over ``tasks``."""
        opt = torch.optim.SGD(model.parameters(), lr=self.meta_lr)
        opt.zero_grad()
        total_loss = 0.0
        for data in tasks:
            clone = self.adapt(model, data)
            loss = self.train_fn(clone, data)
            loss.backward()
            for bp, cp in zip(model.parameters(), clone.parameters()):
                if cp.grad is not None:
                    if bp.grad is None:
                        bp.grad = cp.grad.detach().clone()
                    else:
                        bp.grad.add_(cp.grad.detach())
            total_loss += float(loss)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(len(tasks))
        opt.step()
        return total_loss / len(tasks)


__all__ = ["MetaOptimizer"]
