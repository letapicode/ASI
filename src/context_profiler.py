"""Measure memory usage and time for various context lengths."""

from __future__ import annotations

import time
import torch

from .hierarchical_memory import HierarchicalMemory
from .telemetry import FineGrainedProfiler


def profile_model(model: torch.nn.Module, lengths: list[int]) -> list[dict[str, float]]:
    """Run ``model`` on random sequences of different lengths."""
    device = next(model.parameters()).device
    results = []
    for L in lengths:
        x = torch.randint(0, 100, (1, L), device=device)
        stats: dict[str, float] = {"length": L}

        def cb(cpu, gpu):
            stats["cpu_time"] = cpu
            stats["gpu_mem"] = gpu

        with FineGrainedProfiler(cb):
            model(x)
        results.append(stats)
    return results


__all__ = ["profile_model"]
