from __future__ import annotations

"""Queue HPC jobs when carbon intensity is low."""

from dataclasses import dataclass, field
import time
from typing import List, Optional, Union, Any

import requests

if not hasattr(requests, "get"):
    requests.get = lambda *a, **k: None  # type: ignore

from .telemetry import TelemetryLogger
from .hpc_schedulers import HPCJobScheduler, submit_job


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


def _default_tracker() -> "CarbonFootprintTracker":  # pragma: no cover - fallback
    try:
        from .carbon_tracker import CarbonFootprintTracker
        return CarbonFootprintTracker(interval=1.0)
    except Exception:
        class _Dummy:
            def start(self) -> None: ...
            def stop(self) -> None: ...

        return _Dummy()  # type: ignore


@dataclass
class CarbonAwareScheduler(HPCJobScheduler):
    """Dispatch jobs when carbon intensity drops below a threshold."""

    max_intensity: float | None = None
    threshold: float | None = None
    region: Optional[str] = None
    telemetry: Optional[TelemetryLogger] = None
    check_interval: float = 60.0
    carbon_api: Optional[str] = None
    backend: str = "slurm"
    tracker: Any = field(default_factory=_default_tracker)

    def __post_init__(self) -> None:
        if self.max_intensity is None:
            self.max_intensity = self.threshold
        if self.max_intensity is None:
            raise ValueError("max_intensity must be specified")
        super().__init__(
            backend=self.backend,
            telemetry=self.telemetry,
            region=self.region,
            max_intensity=self.max_intensity,
            carbon_api=self.carbon_api,
            check_interval=self.check_interval,
        )

    # --------------------------------------------------
    def submit_when_green(self, command: Union[str, List[str]]) -> str:
        """Submit ``command`` once intensity drops below ``max_intensity``."""
        self.tracker.start()
        try:
            while self._fetch_intensity() > self.max_intensity:
                time.sleep(self.check_interval)
            job_id = submit_job(
                command,
                backend=self.backend,
                telemetry=self.telemetry,
                region=self.region,
                max_intensity=self.max_intensity,
                carbon_api=self.carbon_api,
            )
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
            job_id = submit_job(
                command,
                backend=self.backend,
                telemetry=self.telemetry,
                region=self.region,
                max_intensity=self.max_intensity,
                carbon_api=self.carbon_api,
            )
        finally:
            self.tracker.stop()
        return job_id


__all__ = [
    "get_carbon_intensity",
    "get_hourly_forecast",
    "CarbonAwareScheduler",
]
