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


def get_current_price(provider: str, region: str, instance_type: str) -> float:
    """Return current energy price for ``provider`` and ``region``."""
    if provider.lower() == "aws":
        url = f"https://pricing.aws.com/{region}/{instance_type}/price"
    elif provider.lower() in {"gcp", "google"}:
        url = f"https://cloudpricing.googleapis.com/{region}/{instance_type}/price"
    elif provider.lower() in {"azure", "az"}:
        url = f"https://azurepricing.microsoft.com/{region}/{instance_type}/price"
    else:
        raise ValueError(f"Unknown provider {provider}")
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("price", 0.0))
    except Exception:
        return 0.0


def get_hourly_price_forecast(provider: str, region: str, instance_type: str) -> List[float]:
    """Return a 24h price forecast in $/kWh."""
    if provider.lower() == "aws":
        url = f"https://pricing.aws.com/{region}/{instance_type}/forecast"
    elif provider.lower() in {"gcp", "google"}:
        url = f"https://cloudpricing.googleapis.com/{region}/{instance_type}/forecast"
    elif provider.lower() in {"azure", "az"}:
        url = f"https://azurepricing.microsoft.com/{region}/{instance_type}/forecast"
    else:
        raise ValueError(f"Unknown provider {provider}")
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        return [float(p) for p in data.get("forecast", [])]
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


@dataclass
class CostAwareScheduler:
    """Schedule jobs when energy price is low."""

    provider: str = "aws"
    region: str = "us-east-1"
    instance_type: str = "m5.large"
    threshold: float = 0.2
    backend: str = "slurm"
    check_interval: float = 600.0

    def submit_when_cheap(self, command: Union[str, List[str]]) -> str:
        """Submit ``command`` once price falls below ``threshold``."""
        while True:
            price = get_current_price(self.provider, self.region, self.instance_type)
            if price <= self.threshold:
                break
            time.sleep(self.check_interval)
        return submit_job(command, backend=self.backend)

    def submit_at_optimal_time(self, command: Union[str, List[str]], max_delay: float = 21600.0) -> str:
        """Submit at the lowest-price slot within ``max_delay``."""
        forecast = get_hourly_price_forecast(self.provider, self.region, self.instance_type)
        delay = 0.0
        if forecast:
            min_idx = int(min(range(len(forecast)), key=lambda i: forecast[i]))
            delay = min_idx * 3600.0
        if delay and delay <= max_delay:
            time.sleep(delay)
        return submit_job(command, backend=self.backend)


@dataclass
class CarbonCostAwareScheduler(CarbonAwareScheduler):
    """Combine carbon and price forecasts for scheduling."""

    provider: str = "aws"
    instance_type: str = "m5.large"
    carbon_weight: float = 0.5
    cost_weight: float = 0.5

    def submit_at_optimal_time(self, command: Union[str, List[str]], max_delay: float = 21600.0) -> str:
        carbon_forecast = get_hourly_forecast(self.region)
        price_forecast = get_hourly_price_forecast(self.provider, self.region, self.instance_type)
        n = min(len(carbon_forecast), len(price_forecast))
        delay = 0.0
        if n:
            scores = [
                self.carbon_weight * carbon_forecast[i] + self.cost_weight * price_forecast[i]
                for i in range(n)
            ]
            min_idx = int(min(range(n), key=lambda i: scores[i]))
            delay = min_idx * 3600.0
        if delay and delay <= max_delay:
            time.sleep(delay)
        return submit_job(command, backend=self.backend)


@dataclass
class MultiProviderScheduler(CarbonCostAwareScheduler):
    """Select the cheapest-greenest cloud provider."""

    providers: List[str] = field(default_factory=lambda: ["aws", "gcp", "azure"])
    backend_map: dict[str, str] = field(
        default_factory=lambda: {"aws": "slurm", "gcp": "k8s", "azure": "slurm"}
    )

    def submit_at_optimal_time(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> str:
        carbon_forecast = get_hourly_forecast(self.region)
        best_provider = self.provider
        best_score = float("inf")
        best_delay = 0.0
        for prov in self.providers:
            price_forecast = get_hourly_price_forecast(prov, self.region, self.instance_type)
            n = min(len(carbon_forecast), len(price_forecast))
            if not n:
                continue
            scores = [
                self.carbon_weight * carbon_forecast[i] + self.cost_weight * price_forecast[i]
                for i in range(n)
            ]
            idx = int(min(range(n), key=lambda i: scores[i]))
            if scores[idx] < best_score:
                best_score = scores[idx]
                best_delay = idx * 3600.0
                best_provider = prov
        if best_delay and best_delay <= max_delay:
            time.sleep(best_delay)
        backend = self.backend_map.get(best_provider, self.backend)
        return submit_job(command, backend=backend)

__all__ = [
    "get_carbon_intensity",
    "get_hourly_forecast",
    "get_current_price",
    "get_hourly_price_forecast",
    "CarbonAwareScheduler",
    "CostAwareScheduler",
    "CarbonCostAwareScheduler",
    "MultiProviderScheduler",
]
