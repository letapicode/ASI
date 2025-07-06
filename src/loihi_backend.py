from __future__ import annotations

import torch
from dataclasses import dataclass

try:
    import nxsdk.api.n2a as nx  # type: ignore
    _HAS_LOIHI = True
except Exception:  # pragma: no cover - optional dependency
    nx = None  # type: ignore
    _HAS_LOIHI = False


@dataclass
class LoihiConfig:
    """Configuration options for Loihi inference."""

    num_cores: int = 1
    spike_precision: int = 8


_CONFIG = LoihiConfig()


def configure_loihi(config: LoihiConfig) -> None:
    """Set the global Loihi configuration."""
    global _CONFIG
    _CONFIG = config


def get_loihi_config() -> LoihiConfig:
    return _CONFIG


def lif_forward(
    mem: torch.Tensor,
    x: torch.Tensor,
    decay: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an LIF step on Loihi if available."""
    if _HAS_LOIHI and hasattr(nx, "lif_forward"):
        cfg = get_loihi_config()
        return nx.lif_forward(
            mem,
            x,
            decay=decay,
            threshold=threshold,
            num_cores=cfg.num_cores,
            precision=cfg.spike_precision,
        )  # type: ignore
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
        cfg = get_loihi_config()
        return nx.linear_forward(
            x,
            weight,
            bias=bias,
            num_cores=cfg.num_cores,
            precision=cfg.spike_precision,
        )  # type: ignore
    return torch.nn.functional.linear(x, weight, bias)


__all__ = [
    "_HAS_LOIHI",
    "LoihiConfig",
    "configure_loihi",
    "get_loihi_config",
    "lif_forward",
    "linear_forward",
]

