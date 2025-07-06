import subprocess
import json
import urllib.request
from typing import List, Union, Optional

from .telemetry import TelemetryLogger


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


__all__ = ["submit_job", "monitor_job", "cancel_job"]
