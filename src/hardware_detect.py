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

from . import hardware_backends as backends

# cache (devices, env_str, sim_id)
_ANALOG_DEVICES_CACHE: tuple[list[str], str | None, int] | None = None


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
    if getattr(backends, "_HAS_FPGA", False) and getattr(backends, "cl", None):
        try:
            devices: list[str] = []
            for platform in backends.cl.get_platforms():
                for dev in platform.get_devices(device_type=backends.cl.device_type.ACCELERATOR):
                    devices.append(getattr(dev, "name", "fpga"))
            return devices or ["fpga0"]
        except Exception:
            return ["fpga0"]
    return []


def list_loihi() -> list[str]:
    """Return a list of available Loihi accelerator identifiers."""
    if getattr(backends, "_HAS_LOIHI", False):
        return ["loihi0"]
    return []


def list_analog() -> list[str]:
    """Return a list of available analog accelerator identifiers."""
    global _ANALOG_DEVICES_CACHE
    env = os.getenv("ASI_ANALOG_DEVICES")
    sim = getattr(backends, "analogsim", None)
    key = (env, id(sim))
    if _ANALOG_DEVICES_CACHE is not None and _ANALOG_DEVICES_CACHE[1:] == key:
        return list(_ANALOG_DEVICES_CACHE[0])

    if not getattr(backends, "_HAS_ANALOG", False):
        _ANALOG_DEVICES_CACHE = ([], env, id(sim))
        return []

    if env:
        devices = [d.strip() for d in env.split(",") if d.strip()]
        if devices:
            _ANALOG_DEVICES_CACHE = (devices, env, id(sim))
            return list(devices)
    if sim is not None:
        if hasattr(sim, "list_devices"):
            try:
                devs = sim.list_devices()
                if devs:
                    devices = [str(d) for d in devs]
                    _ANALOG_DEVICES_CACHE = (devices, env, id(sim))
                    return list(devices)
            except Exception:
                pass
        if hasattr(sim, "device_count"):
            try:
                count = int(sim.device_count())
                devices = [f"analog{i}" for i in range(count)]
                _ANALOG_DEVICES_CACHE = (devices, env, id(sim))
                return list(devices)
            except Exception:
                pass

    _ANALOG_DEVICES_CACHE = (["analog0"], env, id(sim))
    return ["analog0"]


__all__ = [
    "list_cpus",
    "list_gpus",
    "list_fpgas",
    "list_loihi",
    "list_analog",
]
