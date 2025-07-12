from __future__ import annotations

"""Common utilities for HPC job scheduling."""

import json
import subprocess
import threading
import time
import urllib.request
from collections import deque
from typing import Deque, List, Optional, Union

try:  # pragma: no cover - optional psutil
    import psutil  # type: ignore
except Exception:  # pragma: no cover - allow running without psutil
    psutil = None  # type: ignore

from .telemetry import TelemetryLogger
from dataclasses import dataclass, field
from typing import Dict, Protocol, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints
    from .dashboards import ClusterCarbonDashboard



# ---------------------------------------------------------------------------
def submit_job(
    command: Union[str, List[str]],
    backend: str = "slurm",
    *,
    telemetry: Optional[TelemetryLogger] = None,
    region: Optional[str] = None,
    max_intensity: Optional[float] = None,
    max_temp: Optional[float] = None,
    carbon_api: Optional[str] = None,
) -> str:
    """Submit a job if carbon intensity and GPU temperature permit."""
    if telemetry is not None and max_intensity is not None:
        intensity = telemetry.get_carbon_intensity(region)
        if intensity > max_intensity:
            return "DEFERRED"
    if telemetry is not None and max_temp is not None:
        try:
            if telemetry.gpu_temperature() >= max_temp:
                return "DEFERRED"
        except Exception:
            pass
    if carbon_api and max_intensity is not None:
        try:
            with urllib.request.urlopen(carbon_api, timeout=1) as r:
                data = json.loads(r.read().decode() or "{}")
                intensity = float(data.get("carbon_intensity", 0.0))
                if intensity > max_intensity:
                    return "DEFERRED"
        except Exception:
            pass
    if isinstance(command, str):
        command = [command]
    if backend == "slurm":
        cmd = ["sbatch", *command]
    elif backend in {"kubernetes", "k8s"}:
        cmd = ["kubectl", "apply", "-f", *command]
    else:
        raise ValueError(f"Unknown backend {backend}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if hasattr(proc, "check_returncode"):
        proc.check_returncode()
    elif proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
def monitor_job(job_id: str, backend: str = "slurm") -> str:
    """Return scheduler status for ``job_id``."""
    if backend == "slurm":
        cmd = ["squeue", "-h", "-j", str(job_id), "-o", "%T"]
    elif backend in {"kubernetes", "k8s"}:
        cmd = ["kubectl", "get", "job", job_id, "-o", "jsonpath={.status.succeeded}"]
    else:
        raise ValueError(f"Unknown backend {backend}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if hasattr(proc, "check_returncode"):
        proc.check_returncode()
    elif proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
def cancel_job(job_id: str, backend: str = "slurm") -> str:
    """Cancel a running job."""
    if backend == "slurm":
        cmd = ["scancel", str(job_id)]
    elif backend in {"kubernetes", "k8s"}:
        cmd = ["kubectl", "delete", "job", job_id]
    else:
        raise ValueError(f"Unknown backend {backend}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if hasattr(proc, "check_returncode"):
        proc.check_returncode()
    elif proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
class HPCJobScheduler:
    """Base scheduler with queueing and environmental checks."""

    def __init__(
        self,
        *,
        backend: str = "slurm",
        telemetry: Optional[TelemetryLogger] = None,
        region: Optional[str] = None,
        max_intensity: Optional[float] = None,
        carbon_api: Optional[str] = None,
        battery_threshold: Optional[float] = None,
        check_interval: float = 60.0,
    ) -> None:
        self.backend = backend
        self.telemetry = telemetry or TelemetryLogger(interval=check_interval)
        self.region = region
        self.max_intensity = max_intensity
        self.carbon_api = carbon_api
        self.battery_threshold = battery_threshold
        self.check_interval = check_interval
        self.queue: Deque[Union[str, List[str]]] = deque()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # --------------------------------------------------
    def _fetch_intensity(self) -> float:
        if self.carbon_api:
            try:
                with urllib.request.urlopen(self.carbon_api, timeout=1) as r:
                    data = json.loads(r.read().decode() or "{}")
                    return float(data.get("carbon_intensity", 0.0))
            except Exception:
                pass
        return self.telemetry.get_carbon_intensity(self.region)

    # --------------------------------------------------
    def _carbon_ok(self) -> bool:
        if self.max_intensity is None:
            return True
        return self._fetch_intensity() <= self.max_intensity

    # --------------------------------------------------
    def _battery_ok(self) -> bool:
        if self.battery_threshold is None or psutil is None:
            return True
        try:
            info = psutil.sensors_battery()
            if info is not None and info.percent is not None:
                level = float(info.percent) / 100.0
                self.telemetry.metrics["battery"] = info.percent
                return level >= self.battery_threshold
        except Exception:
            pass
        return True

    # --------------------------------------------------
    def _can_run(self) -> bool:
        return self._carbon_ok() and self._battery_ok()

    # --------------------------------------------------
    def queue_job(self, command: Union[str, List[str]]) -> None:
        self.queue.append(command)

    # --------------------------------------------------
    def submit_job(self, command: Union[str, List[str]], backend: Optional[str] = None) -> str:
        if self._can_run():
            return submit_job(
                command,
                backend=backend or self.backend,
                telemetry=self.telemetry,
                region=self.region,
                max_intensity=self.max_intensity,
                carbon_api=self.carbon_api,
            )
        self.queue_job(command)
        return "QUEUED"

    # --------------------------------------------------
    def monitor_job(self, job_id: str, backend: Optional[str] = None) -> str:
        return monitor_job(job_id, backend=backend or self.backend)

    # --------------------------------------------------
    def cancel_job(self, job_id: str, backend: Optional[str] = None) -> str:
        return cancel_job(job_id, backend=backend or self.backend)

    # --------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.queue and self._can_run():
                cmd = self.queue.popleft()
                submit_job(
                    cmd,
                    backend=self.backend,
                    telemetry=self.telemetry,
                    region=self.region,
                    max_intensity=self.max_intensity,
                    carbon_api=self.carbon_api,
                )
            else:
                time.sleep(self.check_interval)

    # --------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
class ForecastStrategy(Protocol):
    """Interface for forecasting cluster cost/carbon scores."""

    def forecast_scores(
        self,
        scheduler: "HPCBaseScheduler",
        max_delay: float,
        clusters: Dict[str, "HPCBaseScheduler"] | None = None,
    ) -> List[float]:
        ...


@dataclass
class HPCBaseScheduler:
    """Manage job queueing and submission using a forecasting strategy."""

    carbon_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)
    carbon_weight: float = 0.5
    cost_weight: float = 0.5
    backend: str = "slurm"
    strategy: ForecastStrategy | None = field(default=None, repr=False)

    _queue: List[Union[str, List[str]]] = field(default_factory=list, init=False, repr=False)

    # --------------------------------------------------
    def forecast_scores(
        self, max_delay: float, clusters: Dict[str, "HPCBaseScheduler"] | None = None
    ) -> List[float]:
        """Delegate forecasting to the attached strategy."""
        if self.strategy is None:
            raise NotImplementedError("No forecasting strategy configured")
        return self.strategy.forecast_scores(self, max_delay, clusters)

    # --------------------------------------------------
    def submit_at_optimal_time(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> str:
        """Submit ``command`` when the forecast is most favourable."""
        scores = self.forecast_scores(max_delay)
        delay = 0.0
        if scores:
            idx = int(min(range(len(scores)), key=lambda i: scores[i]))
            delay = idx * 3600.0
        if delay and delay <= max_delay:
            time.sleep(delay)
        return submit_job(command, backend=self.backend)

    # --------------------------------------------------
    def queue_job(self, command: Union[str, List[str]]) -> None:
        """Add a command to the local queue."""
        self._queue.append(command)

    # --------------------------------------------------
    def run_queue(self, max_delay: float = 21600.0) -> List[str]:
        """Submit all queued commands sequentially."""
        results: List[str] = []
        while self._queue:
            cmd = self._queue.pop(0)
            results.append(self.submit_at_optimal_time(cmd, max_delay))
        return results


def make_scheduler(strategy_name: str, **kw) -> HPCBaseScheduler:
    """Return :class:`HPCBaseScheduler` with the chosen forecasting strategy."""

    name = strategy_name.lower()
    strat_kw: Dict[str, object] = {}
    if name == "gnn" and "hist_len" in kw:
        strat_kw["hist_len"] = kw.pop("hist_len")

    sched = HPCBaseScheduler(**kw)
    if name == "arima":
        from .forecast_strategies import ArimaStrategy

        sched.strategy = ArimaStrategy()
    elif name == "gnn":
        from .forecast_strategies import GNNStrategy

        sched.strategy = GNNStrategy(**strat_kw)
    else:
        raise ValueError(f"Unknown strategy {strategy_name}")
    return sched


def _record_carbon_saving(
    telemetry_map: Optional[Dict[str, TelemetryLogger]],
    tel: Optional[TelemetryLogger],
    cluster: str,
    duration: float,
    log: List[Tuple[str, float]],
    dashboard: Optional["ClusterCarbonDashboard"],
) -> None:
    """Record carbon savings for a scheduled job."""
    if telemetry_map and tel is not None and len(telemetry_map) > 0:
        baseline = sum(
            t.get_live_carbon_intensity() for t in telemetry_map.values()
        ) / len(telemetry_map)
        chosen = tel.get_live_carbon_intensity()
        saving = (baseline - chosen) * duration
        tel.metrics["carbon_saved"] = tel.metrics.get("carbon_saved", 0.0) + saving
        log.append((cluster, saving))
        if dashboard is not None:
            dashboard.record_schedule(cluster, saving)


@dataclass
class MultiClusterScheduler:
    """Compare forecasts from multiple clusters and submit to the best one."""

    clusters: Dict[str, HPCBaseScheduler] = field(default_factory=dict)
    telemetry: Optional[Dict[str, TelemetryLogger]] = None
    dashboard: Optional["ClusterCarbonDashboard"] = None
    schedule_log: list[tuple[str, float]] = field(default_factory=list)

    # --------------------------------------------------
    def submit_best(
        self,
        command: Union[str, List[str]],
        max_delay: float = 21600.0,
        *,
        expected_duration: float = 1.0,
    ) -> Tuple[str, str]:
        """Return chosen cluster name and job id after submission."""

        best_cluster = None
        best_backend = None
        best_score = float("inf")
        best_delay = 0.0

        for name, sched in self.clusters.items():
            scores = sched.forecast_scores(max_delay, self.clusters)
            if not scores:
                continue
            idx = int(min(range(len(scores)), key=lambda i: scores[i]))
            if scores[idx] < best_score:
                best_score = scores[idx]
                best_delay = idx * 3600.0
                best_cluster = name
                best_backend = sched.backend

        if best_cluster is None:
            raise ValueError("No forecasts available to choose a cluster")
        if best_delay and best_delay <= max_delay:
            time.sleep(best_delay)
        tel = self.telemetry.get(best_cluster) if self.telemetry else None
        job_id = globals()["submit_job"](
            command, backend=best_backend, telemetry=tel
        )
        _record_carbon_saving(
            self.telemetry,
            tel,
            best_cluster,
            expected_duration,
            self.schedule_log,
            self.dashboard,
        )
        return best_cluster, job_id

    # --------------------------------------------------
    def cluster_stats(self) -> Dict[str, Dict[str, float]]:
        """Return metrics from attached telemetry loggers."""
        out: Dict[str, Dict[str, float]] = {}
        if self.telemetry:
            for name, tel in self.telemetry.items():
                out[name] = tel.get_stats()
                if tel.metrics.get("carbon_saved") is not None:
                    out[name]["carbon_saved"] = float(tel.metrics["carbon_saved"])
        return out


__all__ = [
    "submit_job",
    "monitor_job",
    "cancel_job",
    "HPCJobScheduler",
    "HPCBaseScheduler",
    "ForecastStrategy",
    "make_scheduler",
    "MultiClusterScheduler",
    "_record_carbon_saving",
]
