from __future__ import annotations
import torch
from torch import nn
from typing import Sequence


def merge_adapters(model: nn.Module | None, adapters: Sequence[str], weights: Sequence[float]) -> dict:
    """Load LoRA checkpoints and combine them via weighted averaging."""
    assert len(adapters) == len(weights)
    merged: dict[str, torch.Tensor] = {}
    tot = sum(weights)
    for path, w in zip(adapters, weights):
        state = torch.load(path, map_location="cpu")
        for k, v in state.items():
            merged[k] = merged.get(k, torch.zeros_like(v)) + v * (w / tot)
    if model is not None:
        return {k: merged.get(k, v) for k, v in model.state_dict().items()}
    return merged

__all__ = ["merge_adapters"]
