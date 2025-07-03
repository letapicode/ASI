"""Measure memory usage and time for various context lengths."""

from __future__ import annotations

import time
import torch

from .telemetry import FineGrainedProfiler


class ContextWindowProfiler:
    """Profile model memory usage and runtime over sequence lengths."""

    def __init__(self, model: torch.nn.Module, lengths: list[int]) -> None:
        self.model = model.eval()
        self.lengths = list(lengths)

    def run(self) -> list[dict[str, float]]:
        device = next(self.model.parameters()).device
        results: list[dict[str, float]] = []
        for L in self.lengths:
            x = torch.randint(0, 100, (1, L), device=device)
            stats: dict[str, float] = {"length": L}

            def cb(cpu: float, gpu: float) -> None:
                stats["cpu_time"] = cpu
                stats["gpu_mem"] = gpu / (1024 ** 2)

            with torch.no_grad():
                with FineGrainedProfiler(cb):
                    self.model(x)
            results.append(stats)
        return results


def profile_model(model: torch.nn.Module, lengths: list[int]) -> list[dict[str, float]]:
    """Legacy helper wrapping :class:`ContextWindowProfiler`."""

    return ContextWindowProfiler(model, lengths).run()


__all__ = ["ContextWindowProfiler", "profile_model"]
