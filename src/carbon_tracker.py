from __future__ import annotations

"""Energy and carbon footprint tracking."""

from dataclasses import dataclass, field
import threading
import time
from typing import Dict

try:  # pragma: no cover - optional
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:  # pragma: no cover - graceful fallback
    _HAS_NVML = False

import psutil


@dataclass
class CarbonFootprintTracker:
    """Track energy usage and carbon emissions."""

    interval: float = 1.0
    co2_per_kwh: float = 400.0
    cpu_tdp: float = 65.0
    metrics: Dict[str, float] = field(default_factory=lambda: {
        "energy_kwh": 0.0,
        "carbon_g": 0.0,
    })

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None
        if _HAS_NVML:
            self.handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
        else:  # pragma: no cover - no GPUs detected
            self.handles = []

    # --------------------------------------------------------------
    def _sample_power(self) -> float:
        cpu_power = psutil.cpu_percent(interval=None) / 100.0 * self.cpu_tdp
        gpu_power = 0.0
        if _HAS_NVML:
            for h in self.handles:
                try:
                    gpu_power += pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:  # pragma: no cover - skip failures
                    pass
        return cpu_power + gpu_power

    def _collect(self) -> None:
        while not self._stop.is_set():
            p = self._sample_power()
            self.metrics["energy_kwh"] += p * self.interval / 3600.0 / 1000.0
            self.metrics["carbon_g"] = (
                self.metrics["energy_kwh"] * self.co2_per_kwh
            )
            time.sleep(self.interval)

    # --------------------------------------------------------------
    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.thread is None:
            return
        self._stop.set()
        self.thread.join(timeout=1.0)
        self.thread = None
        self._stop.clear()

    def get_stats(self) -> Dict[str, float]:
        return dict(self.metrics)


__all__ = ["CarbonFootprintTracker"]
