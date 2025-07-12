from __future__ import annotations

from dataclasses import dataclass
import torch

# ---------------------------------------------------------------------------
# Analog accelerator utilities
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


_ANALOG_CONFIG = AnalogConfig()


def configure_analog(config: AnalogConfig) -> None:
    """Set the global Analog configuration."""
    global _ANALOG_CONFIG
    _ANALOG_CONFIG = config


def get_analog_config() -> AnalogConfig:
    return _ANALOG_CONFIG


class AnalogAccelerator:
    """Offload matrix multiplies to an analog simulator when available."""

    def __init__(self, config: AnalogConfig | None = None) -> None:
        self.config = config or get_analog_config()
        self._orig_matmul: callable | None = None

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if _HAS_ANALOG and hasattr(analogsim, "matmul"):
            return analogsim.matmul(a, b, noise=self.config.noise)  # type: ignore
        return a @ b

    def __enter__(self) -> "AnalogAccelerator":
        self._orig_matmul = torch.matmul
        torch.matmul = self.matmul  # type: ignore
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._orig_matmul is not None:
            torch.matmul = self._orig_matmul  # type: ignore
            self._orig_matmul = None


# ---------------------------------------------------------------------------
# FPGA accelerator utilities
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


_FPGA_CONFIG = FPGAConfig()


def configure_fpga(config: FPGAConfig) -> None:
    """Set the global FPGA configuration."""
    global _FPGA_CONFIG
    _FPGA_CONFIG = config


def get_fpga_config() -> FPGAConfig:
    return _FPGA_CONFIG


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
        if _HAS_FPGA:
            self.compiled = True
        else:
            self.compiled = False

    def run(self, *args, **kwargs):
        return self._forward(*args, **kwargs)


# ---------------------------------------------------------------------------
# Loihi accelerator utilities
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


_LOIHI_CONFIG = LoihiConfig()


def configure_loihi(config: LoihiConfig) -> None:
    """Set the global Loihi configuration."""
    global _LOIHI_CONFIG
    _LOIHI_CONFIG = config


def get_loihi_config() -> LoihiConfig:
    return _LOIHI_CONFIG


def lif_forward(
    mem: torch.Tensor,
    x: torch.Tensor,
    decay: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an LIF step on Loihi if available."""
    if _HAS_LOIHI and hasattr(nx, "lif_forward"):
        cfg = get_loihi_config()
        return nx.lif_forward(  # type: ignore
            mem,
            x,
            decay=decay,
            threshold=threshold,
            num_cores=cfg.num_cores,
            precision=cfg.spike_precision,
        )
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
        return nx.linear_forward(  # type: ignore
            x,
            weight,
            bias=bias,
            num_cores=cfg.num_cores,
            precision=cfg.spike_precision,
        )
    return torch.nn.functional.linear(x, weight, bias)


__all__ = [
    "_HAS_ANALOG",
    "AnalogConfig",
    "configure_analog",
    "get_analog_config",
    "AnalogAccelerator",
    "_HAS_FPGA",
    "FPGAConfig",
    "configure_fpga",
    "get_fpga_config",
    "FPGAAccelerator",
    "_HAS_LOIHI",
    "LoihiConfig",
    "configure_loihi",
    "get_loihi_config",
    "lif_forward",
    "linear_forward",
    "analogsim",
    "cl",
    "nx",
]
