from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch


@dataclass
class Sim2RealParams:
    """Simple bias parameters learned from real logs."""

    bias: torch.Tensor


def learn_env_params(
    logs: Iterable[Tuple[torch.Tensor, torch.Tensor]]
) -> Sim2RealParams:
    """Estimate observation bias from ``(sim, real)`` pairs."""
    diffs = [real - sim for sim, real in logs]
    if not diffs:
        return Sim2RealParams(bias=torch.zeros(0))
    bias = torch.stack(diffs).mean(dim=0)
    return Sim2RealParams(bias=bias)


def apply_correction(
    transitions: Iterable[tuple[torch.Tensor, int, torch.Tensor, float]],
    params: Sim2RealParams,
) -> List[tuple[torch.Tensor, int, torch.Tensor, float]]:
    """Add learned bias to states in ``transitions``."""
    corrected = []
    for state, action, next_state, reward in transitions:
        if params.bias.numel() == state.numel():
            state = state + params.bias
            next_state = next_state + params.bias
        corrected.append((state, action, next_state, reward))
    return corrected


__all__ = ["Sim2RealParams", "learn_env_params", "apply_correction"]
