from __future__ import annotations

"""Queue HPC jobs based on regional carbon intensity."""

import threading
import time
import json
import urllib.request
from collections import deque
from typing import Deque, List, Union, Optional, Tuple

from .telemetry import TelemetryLogger
from .hpc_scheduler import submit_job, monitor_job, cancel_job


class CarbonAwareScheduler:
    """Dispatch jobs when carbon intensity drops below a threshold."""

    def __init__(
        self,
        max_intensity: float,
        *,
        region: Optional[str] = None,
        telemetry: Optional[TelemetryLogger] = None,
        check_interval: float = 60.0,
        carbon_api: Optional[str] = None,
    ) -> None:
        self.max_intensity = max_intensity
        self.region = region
        self.carbon_api = carbon_api
        self.telemetry = telemetry or TelemetryLogger(interval=check_interval)
        self.check_interval = check_interval
        self.queue: Deque[Tuple[Union[str, List[str]], str]] = deque()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # --------------------------------------------------------------
    def _fetch_intensity(self) -> float:
        if self.carbon_api:
            try:
                with urllib.request.urlopen(self.carbon_api, timeout=1) as r:
                    data = json.loads(r.read().decode() or "{}")
                    return float(data.get("carbon_intensity", 0.0))
            except Exception:
                pass
        return self.telemetry.get_carbon_intensity(self.region)

    # --------------------------------------------------------------
    def submit_job(self, command: Union[str, List[str]], backend: str = "slurm") -> str:
        intensity = self._fetch_intensity()
        if intensity <= self.max_intensity:
            return submit_job(
                command,
                backend=backend,
                telemetry=self.telemetry,
                region=self.region,
                max_intensity=self.max_intensity,
                carbon_api=self.carbon_api,
            )
        self.queue.append((command, backend))
        return "QUEUED"

    # --------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.queue and self._fetch_intensity() <= self.max_intensity:
                command, backend = self.queue.popleft()
                submit_job(
                    command,
                    backend=backend,
                    telemetry=self.telemetry,
                    region=self.region,
                    max_intensity=self.max_intensity,
                    carbon_api=self.carbon_api,
                )
            else:
                time.sleep(self.check_interval)

    # --------------------------------------------------------------
    def monitor_job(self, job_id: str, backend: str = "slurm") -> str:
        return monitor_job(job_id, backend=backend)

    def cancel_job(self, job_id: str, backend: str = "slurm") -> str:
        return cancel_job(job_id, backend=backend)

    # --------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


__all__ = ["CarbonAwareScheduler"]
