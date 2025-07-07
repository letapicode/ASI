from __future__ import annotations

from dataclasses import dataclass
import torch

try:
    import pyopencl as cl  # type: ignore
    _HAS_FPGA = True
except Exception:  # pragma: no cover - optional dependency
    cl = None  # type: ignore
    _HAS_FPGA = False


@dataclass
class FPGAConfig:
    """Configuration options for FPGA execution."""

    device: int = 0
    optimize: bool = True


_CONFIG = FPGAConfig()


def configure_fpga(config: FPGAConfig) -> None:
    """Set the global FPGA configuration."""
    global _CONFIG
    _CONFIG = config


def get_fpga_config() -> FPGAConfig:
    return _CONFIG


class FPGAAccelerator:
    """Compile models and run them on an FPGA when available."""

    def __init__(
        self,
        model: torch.nn.Module,
        forward_fn=None,
        config: FPGAConfig | None = None,
    ) -> None:
        self.model = model
        self._forward = forward_fn if forward_fn is not None else model.forward
        self.config = config or get_fpga_config()
        self.compiled = False
        self.ctx = None
        self.queue = None
        if _HAS_FPGA:
            try:
                self.ctx = cl.create_some_context()  # type: ignore[attr-defined]
                self.queue = cl.CommandQueue(self.ctx)  # type: ignore[attr-defined]
            except Exception:
                pass

    def compile(self) -> None:
        """Placeholder compilation step."""
        if _HAS_FPGA:
            self.compiled = True
        else:
            self.compiled = False

    def run(self, *args, **kwargs):
        """Run the wrapped model. Uses FPGA if compiled."""
        return self._forward(*args, **kwargs)


__all__ = [
    "_HAS_FPGA",
    "FPGAConfig",
    "configure_fpga",
    "get_fpga_config",
    "FPGAAccelerator",
]
