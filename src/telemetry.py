"""Telemetry logging utilities."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Callable

import psutil
from .carbon_tracker import CarbonFootprintTracker
try:  # pragma: no cover - optional torch dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - allow running without torch
    class _DummyCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def utilization() -> float:
            return 0.0

        @staticmethod
        def memory_allocated() -> int:
            return 0

    torch = type("torch", (), {"cuda": _DummyCuda})()  # type: ignore
try:
    from prometheus_client import Gauge, start_http_server
    _HAS_PROM = True
except Exception:  # pragma: no cover - optional
    _HAS_PROM = False


@dataclass
class TelemetryLogger:
    """Collect and export basic hardware metrics."""

    interval: float = 1.0
    port: int | None = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    carbon_tracker: CarbonFootprintTracker | None = None

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None
        if isinstance(self.carbon_tracker, bool):
            if self.carbon_tracker:
                self.carbon_tracker = CarbonFootprintTracker(interval=self.interval)
            else:
                self.carbon_tracker = None
        if _HAS_PROM and self.port is not None:
            start_http_server(self.port)
        if _HAS_PROM:
            self.metrics = {
                "cpu": Gauge("cpu_util", "CPU utilisation"),
                "gpu": Gauge("gpu_util", "GPU utilisation"),
                "mem": Gauge("mem_used", "RAM usage"),
                "net": Gauge("net_sent", "Network bytes sent"),
            }
        else:
            self.metrics = {}

    # --------------------------------------------------------------
    def _collect(self) -> None:
        last_net = psutil.net_io_counters()
        while not self._stop.is_set():
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            gpu = (
                torch.cuda.utilization() if torch.cuda.is_available() else 0.0
            )
            net = psutil.net_io_counters()
            sent = net.bytes_sent - last_net.bytes_sent
            last_net = net
            if self.carbon_tracker is not None:
                cf_stats = self.carbon_tracker.get_stats()
            else:
                cf_stats = {}

            if _HAS_PROM:
                self.metrics["cpu"].set(cpu)
                self.metrics["gpu"].set(gpu)
                self.metrics["mem"].set(mem)
                self.metrics["net"].set(sent)
                if self.carbon_tracker is not None:
                    self.metrics.setdefault("energy_kwh", Gauge("energy_kwh", "Energy consumed"))
                    self.metrics.setdefault("carbon_g", Gauge("carbon_g", "Carbon emitted"))
                    self.metrics["energy_kwh"].set(cf_stats.get("energy_kwh", 0.0))
                    self.metrics["carbon_g"].set(cf_stats.get("carbon_g", 0.0))
            else:
                self.metrics = {
                    "cpu": cpu,
                    "gpu": gpu,
                    "mem": mem,
                    "net": sent,
                }
                self.metrics.update(cf_stats)
            time.sleep(self.interval)

    # --------------------------------------------------------------
    def start(self) -> None:
        if self.thread is not None:
            return
        if self.carbon_tracker is not None:
            self.carbon_tracker.start()
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.thread is None:
            return
        self._stop.set()
        self.thread.join(timeout=1.0)
        self.thread = None
        self._stop.clear()
        if self.carbon_tracker is not None:
            self.carbon_tracker.stop()

    def get_stats(self) -> Dict[str, Any]:
        stats = dict(self.metrics)
        if self.carbon_tracker is not None:
            stats.update(self.carbon_tracker.get_stats())
        return stats


class FineGrainedProfiler:
    """Context manager measuring CPU/GPU time for a block."""

    def __init__(self, callback: Callable[[float, float], None]) -> None:
        self.callback = callback

    def __enter__(self) -> None:
        self.cpu_start = time.perf_counter()
        self.gpu_start = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        cpu_time = time.perf_counter() - self.cpu_start
        gpu_mem = (
            torch.cuda.memory_allocated() - self.gpu_start
            if torch.cuda.is_available()
            else 0
        )
        self.callback(cpu_time, gpu_mem)


__all__ = ["TelemetryLogger", "FineGrainedProfiler", "CarbonFootprintTracker"]
