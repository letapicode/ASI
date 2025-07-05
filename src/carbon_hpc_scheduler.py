from __future__ import annotations

"""Carbon-aware HPC job scheduling utilities."""

from dataclasses import dataclass, field
import time
from typing import List, Union

import requests

from .hpc_scheduler import submit_job
from .carbon_tracker import CarbonFootprintTracker


def get_carbon_intensity(region: str | None = None) -> float:
    """Return current carbon intensity (gCO2/kWh) for ``region``."""
    url = "https://api.carbonintensity.org.uk/intensity"
    if region:
        url = f"https://api.carbonintensity.org.uk/regional/{region}"
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        if region:
            return float(data["data"][0]["data"][0]["intensity"]["forecast"])
        return float(data["data"][0]["intensity"]["forecast"])
    except Exception:
        return 400.0


def get_hourly_forecast(region: str | None = None) -> List[float]:
    """Return a 24h carbon intensity forecast."""
    url = "https://api.carbonintensity.org.uk/intensity/fw24h"
    if region:
        url = f"https://api.carbonintensity.org.uk/regional/{region}/fw24h"
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        if region:
            records = data["data"][0]["data"]
            return [float(r["intensity"]["forecast"]) for r in records]
        return [float(r["intensity"]["forecast"]) for r in data["data"]]
    except Exception:
        return []


@dataclass
class CarbonAwareScheduler:
    """Schedule jobs when carbon intensity is low."""

    backend: str = "slurm"
    region: str | None = None
    threshold: float = 300.0
    check_interval: float = 600.0
    tracker: CarbonFootprintTracker = field(
        default_factory=lambda: CarbonFootprintTracker(interval=1.0)
    )

    # --------------------------------------------------
    def submit_when_green(self, command: Union[str, List[str]]) -> str:
        """Submit ``command`` once intensity drops below ``threshold``."""
        self.tracker.start()
        try:
            while True:
                intensity = get_carbon_intensity(self.region)
                if intensity <= self.threshold:
                    break
                time.sleep(self.check_interval)
            job_id = submit_job(command, backend=self.backend)
        finally:
            self.tracker.stop()
        return job_id

    # --------------------------------------------------
    def submit_at_optimal_time(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> str:
        """Submit at the forecasted lowest-intensity slot within ``max_delay``."""
        self.tracker.start()
        try:
            forecast = get_hourly_forecast(self.region)
            delay = 0.0
            if forecast:
                min_idx = int(min(range(len(forecast)), key=lambda i: forecast[i]))
                delay = min_idx * 3600.0
            if delay and delay <= max_delay:
                time.sleep(delay)
            job_id = submit_job(command, backend=self.backend)
        finally:
            self.tracker.stop()
        return job_id


__all__ = [
    "get_carbon_intensity",
    "get_hourly_forecast",
    "CarbonAwareScheduler",
]
