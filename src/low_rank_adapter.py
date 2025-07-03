"""Low-rank adaptation utilities for parameter-efficient fine-tuning."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    """Linear layer augmented with a LoRA-style low-rank adapter."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        if r > 0:
            self.A = nn.Parameter(torch.zeros(r, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = self.base(x)
        if self.r == 0:
            return out
        if self.dropout is not None:
            x = self.dropout(x)
        lora = F.linear(x, self.B @ self.A) * self.scale
        return out + lora


def apply_low_rank_adaptation(
    model: nn.Module,
    target_modules: Sequence[str],
    r: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    """Replace ``target_modules`` with :class:`LowRankLinear` layers.

    Parameters
    ----------
    model:
        The model whose modules will be replaced in-place.
    target_modules:
        Iterable of attribute names to adapt.
    r:
        Rank of the adapters.
    alpha:
        Scaling factor for the adapter output.
    dropout:
        Optional dropout probability.
    """
    modules = list(model.named_modules())
    for name, module in modules:
        for tgt in target_modules:
            if name.split(".")[-1] == tgt and isinstance(module, nn.Linear):
                parent = model
                for attr in name.split(".")[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, tgt, LowRankLinear(module, r=r, alpha=alpha, dropout=dropout))
    return model


__all__ = ["LowRankLinear", "apply_low_rank_adaptation"]
