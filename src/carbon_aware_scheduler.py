from __future__ import annotations

"""Queue HPC jobs based on regional carbon intensity."""

from typing import List, Optional, Union

from .telemetry import TelemetryLogger
from .hpc_schedulers import HPCJobScheduler


class CarbonAwareScheduler(HPCJobScheduler):
    """Dispatch jobs when carbon intensity drops below a threshold."""

    def __init__(
        self,
        max_intensity: float,
        *,
        region: Optional[str] = None,
        telemetry: Optional[TelemetryLogger] = None,
        check_interval: float = 60.0,
        carbon_api: Optional[str] = None,
        backend: str = "slurm",
    ) -> None:
        super().__init__(
            backend=backend,
            telemetry=telemetry,
            region=region,
            max_intensity=max_intensity,
            carbon_api=carbon_api,
            check_interval=check_interval,
        )


__all__ = ["CarbonAwareScheduler"]
