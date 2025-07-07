from __future__ import annotations

import os

try:  # optional
    import psutil
except Exception:  # pragma: no cover - missing dependency
    psutil = None  # type: ignore

try:  # optional torch dependency
    import torch
except Exception:  # pragma: no cover - allow running without torch
    torch = None  # type: ignore

from . import fpga_backend, analog_backend, loihi_backend


def list_cpus() -> list[str]:
    """Return a list of available CPU identifiers."""
    if psutil is not None:
        count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    else:
        count = os.cpu_count() or 1
    return [f"cpu{i}" for i in range(int(count))]


def list_gpus() -> list[str]:
    """Return a list of available GPU identifiers."""
    if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return []


def list_fpgas() -> list[str]:
    """Return a list of available FPGA device names."""
    if getattr(fpga_backend, "_HAS_FPGA", False) and getattr(fpga_backend, "cl", None):
        try:
            devices: list[str] = []
            for platform in fpga_backend.cl.get_platforms():
                for dev in platform.get_devices(device_type=fpga_backend.cl.device_type.ACCELERATOR):
                    devices.append(getattr(dev, "name", "fpga"))
            return devices or ["fpga0"]
        except Exception:
            return ["fpga0"]
    return []


def list_loihi() -> list[str]:
    """Return a list of available Loihi accelerator identifiers."""
    if getattr(loihi_backend, "_HAS_LOIHI", False):
        return ["loihi0"]
    return []


def list_analog() -> list[str]:
    """Return a list of available analog accelerator identifiers."""
    if getattr(analog_backend, "_HAS_ANALOG", False):
        return ["analog0"]
    return []


__all__ = [
    "list_cpus",
    "list_gpus",
    "list_fpgas",
    "list_loihi",
    "list_analog",
]
