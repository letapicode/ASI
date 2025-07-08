"""Telemetry logging utilities."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from typing import Dict, Any, Callable, Optional, List

from typing import Dict, Any, Callable, Optional
import types
import json
import subprocess
import urllib.request


try:  # pragma: no cover - optional dependency
    import psutil
except Exception:  # pragma: no cover - fallback when missing
    psutil = None  # type: ignore[misc]
try:  # pragma: no cover - optional dependency
    from .carbon_tracker import CarbonFootprintTracker
except Exception:
    CarbonFootprintTracker = None  # type: ignore[misc]
try:
    from .memory_event_detector import MemoryEventDetector
except Exception:
    class MemoryEventDetector:  # type: ignore[dead-code]
        pass
try:
    from .fine_grained_profiler import FineGrainedProfiler
except Exception:
    class FineGrainedProfiler:  # type: ignore[dead-code]
        pass
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
    """Collect and export basic hardware metrics with optional carbon lookup."""

    interval: float = 1.0
    port: int | None = None
    region: Optional[str] = None
    carbon_data: Dict[str, float] | None = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    carbon_tracker: CarbonFootprintTracker | None = None
    energy_price: float = 0.1
    energy_price_data: Dict[str, float] | None = None
    carbon_api: str | None = None

    event_detector: MemoryEventDetector = field(default_factory=MemoryEventDetector)
    history: List[Dict[str, float]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    profiler_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    publish_url: str | None = None
    node_id: str | None = None
    _published_energy: float = field(default=0.0, init=False)
    _published_carbon: float = field(default=0.0, init=False)


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
            self.metrics.setdefault(
                "prof_cpu_time", Gauge("prof_cpu_time", "Fine grained CPU time")
            )
            self.metrics.setdefault(
                "prof_gpu_mem", Gauge("prof_gpu_mem", "Fine grained GPU memory")
            )
        else:
            self.metrics = {}
        if self.carbon_data is None:
            self.carbon_data = {"default": 0.4}
        if self.energy_price_data is None:
            self.energy_price_data = {"default": self.energy_price}

    # --------------------------------------------------------------
    def _collect(self) -> None:
        if psutil is None:
            last_net = types.SimpleNamespace(bytes_sent=0)
        else:
            last_net = psutil.net_io_counters()
        while not self._stop.is_set():
            cpu = psutil.cpu_percent(interval=None) if psutil is not None else 0.0
            mem = psutil.virtual_memory().percent if psutil is not None else 0.0
            gpu = (
                torch.cuda.utilization() if torch.cuda.is_available() else 0.0
            )
            battery = self.get_battery_level()
            if psutil is not None:
                net = psutil.net_io_counters()
                sent = net.bytes_sent - last_net.bytes_sent
                last_net = net
            else:
                sent = 0
            if self.carbon_tracker is not None:
                cf_stats = self.carbon_tracker.get_stats()
                e = cf_stats.get("energy_kwh", 0.0)
                c = cf_stats.get("carbon_g", 0.0)
                intensity = c / e if e else self.get_carbon_intensity()
                cost = e * self.get_energy_price()
                cf_stats["carbon_intensity"] = intensity
                cf_stats["energy_cost"] = cost
            else:
                cf_stats = {}

            if _HAS_PROM:
                self.metrics["cpu"].set(cpu)
                self.metrics["gpu"].set(gpu)
                self.metrics["mem"].set(mem)
                self.metrics["net"].set(sent)
                self.metrics.setdefault("battery", Gauge("battery", "Battery level"))
                self.metrics["battery"].set(battery)
                if self.carbon_tracker is not None:
                    self.metrics.setdefault("energy_kwh", Gauge("energy_kwh", "Energy consumed"))
                    self.metrics.setdefault("carbon_g", Gauge("carbon_g", "Carbon emitted"))
                    self.metrics.setdefault("carbon_intensity", Gauge("carbon_intensity", "Carbon intensity"))
                    self.metrics.setdefault("energy_cost", Gauge("energy_cost", "Energy cost"))
                    self.metrics["energy_kwh"].set(cf_stats.get("energy_kwh", 0.0))
                    self.metrics["carbon_g"].set(cf_stats.get("carbon_g", 0.0))
                    self.metrics["carbon_intensity"].set(cf_stats.get("carbon_intensity", 0.0))
                    self.metrics["energy_cost"].set(cf_stats.get("energy_cost", 0.0))
            else:
                self.metrics = {
                    "cpu": cpu,
                    "gpu": gpu,
                    "mem": mem,
                    "net": sent,
                    "battery": battery,
                }
                self.metrics.update(cf_stats)


            snapshot = {
                "cpu": cpu,
                "gpu": gpu,
                "mem": mem,
                "net": sent,
                "battery": battery,
            }
            snapshot.update(cf_stats)
            self.history.append(snapshot)
            events = self.event_detector.update(snapshot)
            if events:
                self.events.extend(events)

            self.publish_carbon()

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
        stats: Dict[str, Any] = {}
        for k, v in self.metrics.items():
            if _HAS_PROM and isinstance(v, Gauge):
                stats[k] = v._value.get()  # type: ignore[attr-defined]
            else:
                stats[k] = v
        if self.carbon_tracker is not None:
            stats.update(self.carbon_tracker.get_stats())
        if self.profiler_stats:
            stats["profiler_stats"] = {k: dict(v) for k, v in self.profiler_stats.items()}
        return stats

    def get_events(self) -> List[Dict[str, Any]]:
        """Return list of detected telemetry events."""
        return list(self.events)

    def reset_events(self) -> None:
        self.events.clear()
        self.event_detector.events.clear()

    def get_carbon_intensity(self, region: Optional[str] = None) -> float:
        """Return carbon intensity (kgCO2/kWh) for the given region."""
        region = region or self.region or "default"
        assert self.carbon_data is not None
        return float(self.carbon_data.get(region, self.carbon_data.get("default", 0.4)))

    def get_energy_price(self, region: Optional[str] = None) -> float:
        """Return energy price ($/kWh) for the given region."""
        region = region or self.region or "default"
        assert self.energy_price_data is not None
        return float(self.energy_price_data.get(region, self.energy_price_data.get("default", self.energy_price)))

    def get_live_carbon_intensity(self, region: Optional[str] = None) -> float:
        """Return current carbon intensity from tracker or external API."""
        if self.carbon_tracker is not None:
            stats = self.carbon_tracker.get_stats()
            e = stats.get("energy_kwh", 0.0)
            c = stats.get("carbon_g", 0.0)
            if e:
                return c / e
        if self.carbon_api:
            try:
                url = f"{self.carbon_api}?region={region or self.region or 'default'}"
                data = json.loads(urllib.request.urlopen(url, timeout=1).read().decode() or "{}")
                if "carbon_intensity" in data:
                    return float(data["carbon_intensity"])
            except Exception:
                pass
        return self.get_carbon_intensity(region)

    def get_cost_index(self, region: Optional[str] = None) -> float:
        """Return cost index combining price and carbon intensity."""
        return self.get_energy_price(region) * self.get_live_carbon_intensity(region)

    # --------------------------------------------------------------
    def gpu_temperature(self, index: int = 0) -> float:
        """Return GPU temperature in Celsius or ``0.0`` if unavailable."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(index),
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            line = out.splitlines()[0]
            return float(line.strip())
        except Exception:
            pass
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            return float(temp)
        except Exception:
            return 0.0

    # --------------------------------------------------------------
    def get_battery_level(self) -> float:
        """Return current battery level percentage."""
        try:
            info = psutil.sensors_battery()
            if info is not None and info.percent is not None:
                return float(info.percent)
        except Exception:
            pass
        return 100.0

    # --------------------------------------------------------------
    def _post(self, data: Dict[str, Any]) -> None:
        if not self.publish_url:
            return
        try:
            req = urllib.request.Request(
                self.publish_url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=1).read()
        except Exception:
            pass

    def publish_carbon(self) -> None:
        if self.carbon_tracker is None or self.node_id is None:
            return
        stats = self.carbon_tracker.get_stats()
        energy = float(stats.get("energy_kwh", 0.0))
        carbon = float(stats.get("carbon_g", 0.0))
        delta_e = energy - self._published_energy
        delta_c = carbon - self._published_carbon
        if delta_e or delta_c:
            intensity = carbon / energy if energy else 0.0
            cost = energy * self.get_energy_price()
            self._post(
                {
                    "node_id": self.node_id,
                    "energy_kwh": delta_e,
                    "carbon_g": delta_c,
                    "carbon_intensity": intensity,
                    "energy_cost": cost,
                }
            )
            self._published_energy = energy
            self._published_carbon = carbon

    # --------------------------------------------------------------
    def register_profiler(
        self, stats: Dict[str, float], node_id: Optional[str] = None
    ) -> None:
        """Record fine-grained metrics from a node."""
        node = node_id or self.node_id or "default"
        entry = self.profiler_stats.setdefault(node, {"cpu_time": 0.0, "gpu_mem": 0.0})
        entry["cpu_time"] += float(stats.get("cpu_time", 0.0))
        entry["gpu_mem"] += float(stats.get("gpu_mem", 0.0))
        cpu_total = sum(v["cpu_time"] for v in self.profiler_stats.values())
        gpu_total = sum(v["gpu_mem"] for v in self.profiler_stats.values())
        if _HAS_PROM:
            assert isinstance(self.metrics["prof_cpu_time"], Gauge)
            assert isinstance(self.metrics["prof_gpu_mem"], Gauge)
            self.metrics["prof_cpu_time"].set(cpu_total)
            self.metrics["prof_gpu_mem"].set(gpu_total)
        else:
            self.metrics["prof_cpu_time"] = cpu_total
            self.metrics["prof_gpu_mem"] = gpu_total





__all__ = [
    "TelemetryLogger",
    "FineGrainedProfiler",
    "CarbonFootprintTracker",
    "MemoryEventDetector",
]
