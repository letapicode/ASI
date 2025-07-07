"""Measure memory usage and time for various context lengths."""

from __future__ import annotations

import time
import torch

from dataclasses import dataclass
from typing import Iterable, List, Dict

from .fine_grained_profiler import FineGrainedProfiler


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


@dataclass
class ContextWindowProfiler:
    """Profile a model across multiple sequence lengths."""

    model: torch.nn.Module

    def profile(self, lengths: Iterable[int]) -> List[Dict[str, float]]:
        """Return runtime stats for each length in ``lengths``."""
        device = next(self.model.parameters()).device
        results: List[Dict[str, float]] = []
        for L in lengths:
            x = torch.randint(0, 100, (1, int(L)), device=device)
            stats: Dict[str, float] = {"length": float(L)}

            def cb(cpu: float, gpu: float) -> None:
                stats["cpu_time"] = cpu
                stats["gpu_mem"] = gpu

            with FineGrainedProfiler(cb):
                self.model(x)
            results.append(stats)
        return results


__all__ = ["profile_model", "ContextWindowProfiler"]
