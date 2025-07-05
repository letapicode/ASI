import subprocess
from typing import List, Union


def submit_job(command: Union[str, List[str]], backend: str = "slurm") -> str:
    """Submit a job via Slurm or Kubernetes."""
    if isinstance(command, str):
        command = [command]
    if backend == "slurm":
        cmd = ["sbatch", *command]
    elif backend in {"kubernetes", "k8s"}:
        cmd = ["kubectl", "apply", "-f", *command]
    else:
        raise ValueError(f"Unknown backend {backend}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    proc.check_returncode()
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
    proc.check_returncode()
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
    proc.check_returncode()
    return proc.stdout.strip()


__all__ = ["submit_job", "monitor_job", "cancel_job"]
