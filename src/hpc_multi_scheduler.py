from __future__ import annotations

"""Select the best HPC cluster based on forecasted cost and carbon intensity."""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Union, Tuple, Optional

from .telemetry import TelemetryLogger
from .dashboards import ClusterCarbonDashboard

from .hpc_base_scheduler import HPCBaseScheduler
from .hpc_schedulers import submit_job


def _record_carbon_saving(
    telemetry_map: Optional[Dict[str, TelemetryLogger]],
    tel: Optional[TelemetryLogger],
    cluster: str,
    duration: float,
    log: List[Tuple[str, float]],
    dashboard: Optional[ClusterCarbonDashboard],
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
    dashboard: Optional[ClusterCarbonDashboard] = None
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
        tel = (
            self.telemetry.get(best_cluster) if self.telemetry else None
        )
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


__all__ = ["MultiClusterScheduler"]
