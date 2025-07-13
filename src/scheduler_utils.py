"""Utility functions for scheduler implementations."""

from __future__ import annotations

from typing import Optional

try:  # optional psutil dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - allow running without psutil
    psutil = None  # type: ignore

try:  # optional torch dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - allow running without torch
    torch = None  # type: ignore

try:  # optional TPU dependency
    import torch_xla.core.xla_model as xm  # type: ignore
    _HAS_XLA = True
except Exception:  # pragma: no cover - allow running without torch_xla
    xm = None  # type: ignore
    _HAS_XLA = False


# --------------------------------------------------------------
def cpu_utilization() -> float:
    """Return current CPU utilization as a ratio [0,1]."""
    if psutil is None:
        return 0.0
    try:
        return psutil.cpu_percent(interval=None) / 100.0
    except Exception:
        return 0.0


# --------------------------------------------------------------
def gpu_utilization(max_temp: Optional[float] = None, telemetry=None) -> float:
    """Return current GPU memory utilization."""
    if torch is not None and torch.cuda.is_available():
        try:
            util = (
                torch.cuda.memory_allocated()
                / torch.cuda.get_device_properties(0).total_memory
            )
        except Exception:
            util = 0.0
        if max_temp is not None and telemetry is not None and util < 1.0:
            try:
                if telemetry.gpu_temperature() >= max_temp:
                    return 1.0
            except Exception:
                pass
        return util
    return 0.0


# --------------------------------------------------------------
def tpu_utilization() -> float:
    """Return current TPU memory utilization."""
    if _HAS_XLA:
        try:
            info = xm.get_memory_info("xla:0")
            used = info.get("kb_total", 0) - info.get("kb_free", 0)
            total = info.get("kb_total", 1)
            return used / total
        except Exception:
            return 0.0
    return 0.0


# --------------------------------------------------------------
def analog_utilization(backend) -> float:
    """Return analog accelerator utilization via backend if available."""
    if getattr(backend, "_HAS_ANALOG", False):
        sim = getattr(backend, "analogsim", None)
        if sim is not None and hasattr(sim, "utilization"):
            try:
                util = sim.utilization()
                return float(util)
            except Exception:
                return 0.0
    return 0.0


# --------------------------------------------------------------
def battery_level() -> float:
    """Return system battery level as a ratio [0,1]."""
    if psutil is None:
        return 1.0
    try:
        info = psutil.sensors_battery()
        if info is not None and info.percent is not None:
            return float(info.percent) / 100.0
    except Exception:
        pass
    return 1.0


__all__ = [
    "psutil",
    "cpu_utilization",
    "gpu_utilization",
    "tpu_utilization",
    "analog_utilization",
    "battery_level",
]
