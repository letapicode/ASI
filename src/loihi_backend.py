from __future__ import annotations

import torch

try:
    import nxsdk.api.n2a as nx  # type: ignore
    _HAS_LOIHI = True
except Exception:  # pragma: no cover - optional dependency
    nx = None  # type: ignore
    _HAS_LOIHI = False


def lif_forward(
    mem: torch.Tensor,
    x: torch.Tensor,
    decay: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an LIF step on Loihi if available."""
    if _HAS_LOIHI and hasattr(nx, "lif_forward"):
        return nx.lif_forward(mem, x, decay=decay, threshold=threshold)  # type: ignore
    mem = mem * decay + x
    spk = (mem >= threshold).to(x.dtype)
    mem = mem - spk * threshold
    return spk, mem


def linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear transform optionally executed on Loihi."""
    if _HAS_LOIHI and hasattr(nx, "linear_forward"):
        return nx.linear_forward(x, weight, bias=bias)  # type: ignore
    return torch.nn.functional.linear(x, weight, bias)


__all__ = ["_HAS_LOIHI", "lif_forward", "linear_forward"]

