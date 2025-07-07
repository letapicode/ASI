from __future__ import annotations

"""Select the best HPC cluster based on forecasted cost and carbon intensity."""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Union, Tuple

from .hpc_forecast_scheduler import arima_forecast, HPCForecastScheduler
from .hpc_gnn_scheduler import GNNForecastScheduler
from .hpc_scheduler import submit_job


@dataclass
class MultiClusterScheduler:
    """Compare forecasts from multiple clusters and submit to the best one."""

    clusters: Dict[str, HPCForecastScheduler] = field(default_factory=dict)

    # --------------------------------------------------
    def submit_best(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> Tuple[str, str]:
        """Return chosen cluster name and job id after submission."""

        best_cluster = None
        best_backend = None
        best_score = float("inf")
        best_delay = 0.0

        for name, sched in self.clusters.items():
            if hasattr(sched, "forecast_scores"):
                scores = sched.forecast_scores(max_delay, self.clusters)
            else:
                steps = max(int(max_delay // 3600) + 1, 1)
                carbon_pred = arima_forecast(sched.carbon_history, steps=steps)
                cost_pred = arima_forecast(sched.cost_history, steps=steps)
                n = min(len(carbon_pred), len(cost_pred))
                scores = [
                    sched.carbon_weight * carbon_pred[i]
                    + sched.cost_weight * cost_pred[i]
                    for i in range(n)
                ]
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
        job_id = globals()["submit_job"](command, backend=best_backend)
        return best_cluster, job_id


__all__ = ["MultiClusterScheduler"]
