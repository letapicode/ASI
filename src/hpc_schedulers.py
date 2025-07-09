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


__all__ = ["submit_job", "monitor_job", "cancel_job", "HPCJobScheduler"]
