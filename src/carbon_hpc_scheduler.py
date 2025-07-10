"""Backward-compatible entry point for carbon-aware HPC scheduling."""

import importlib
import sys

requests = importlib.import_module("requests")
if not hasattr(requests, "get"):
    requests.get = lambda *a, **k: None  # type: ignore
sys.modules.setdefault("requests", requests)

try:  # pragma: no cover - support namespace packages
    from .carbon_aware_scheduler import (
        get_carbon_intensity,
        get_hourly_forecast,
        CarbonAwareScheduler,
    )
except Exception:  # pragma: no cover - fallback when not packaged
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "carbon_aware_scheduler", Path(__file__).with_name("carbon_aware_scheduler.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    sys.modules.setdefault("carbon_aware_scheduler", module)
    get_carbon_intensity = module.get_carbon_intensity  # type: ignore
    get_hourly_forecast = module.get_hourly_forecast  # type: ignore
    CarbonAwareScheduler = module.CarbonAwareScheduler  # type: ignore

__all__ = ["get_carbon_intensity", "get_hourly_forecast", "CarbonAwareScheduler"]
