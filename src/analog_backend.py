from __future__ import annotations

from dataclasses import dataclass
import torch

try:
    import analogsim  # type: ignore
    _HAS_ANALOG = True
except Exception:  # pragma: no cover - optional dependency
    analogsim = None  # type: ignore
    _HAS_ANALOG = False


@dataclass
class AnalogConfig:
    """Configuration options for analog simulation."""

    noise: float = 0.0


_CONFIG = AnalogConfig()


def configure_analog(config: AnalogConfig) -> None:
    """Set the global Analog configuration."""
    global _CONFIG
    _CONFIG = config


def get_analog_config() -> AnalogConfig:
    return _CONFIG


class AnalogAccelerator:
    """Offload matrix multiplies to an analog simulator when available."""

    def __init__(self, config: AnalogConfig | None = None) -> None:
        self.config = config or get_analog_config()
        self._orig_matmul: callable | None = None

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if _HAS_ANALOG and hasattr(analogsim, "matmul"):
            return analogsim.matmul(a, b, noise=self.config.noise)  # type: ignore
        return a @ b

    # --------------------------------------------------
    def __enter__(self) -> "AnalogAccelerator":
        """Monkey patch ``torch.matmul`` within this context."""
        self._orig_matmul = torch.matmul
        torch.matmul = self.matmul  # type: ignore
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._orig_matmul is not None:
            torch.matmul = self._orig_matmul  # type: ignore
            self._orig_matmul = None


__all__ = [
    "_HAS_ANALOG",
    "AnalogConfig",
    "configure_analog",
    "get_analog_config",
    "AnalogAccelerator",
]
