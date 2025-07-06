from __future__ import annotations

"""Schedule HPC jobs based on ARIMA forecasts of cost and carbon intensity."""

from dataclasses import dataclass, field
import time
from typing import List, Union

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .hpc_scheduler import submit_job


# --------------------------------------------------------------
def arima_forecast(series: List[float], steps: int = 1) -> List[float]:
    """Return ARIMA forecast for ``series``."""
    if not series:
        return [0.0] * steps
    try:
        # Simple ARIMA(1,1,0) model; fall back to last value on failure
        order = (1, 1, 0)
        if len(series) < 3:
            order = (0, 0, 0)
        model = ARIMA(series, order=order)
        fit = model.fit()
        forecast = fit.forecast(steps=steps)
        return list(np.asarray(forecast, dtype=float))
    except Exception:
        return [float(series[-1])] * steps


@dataclass
class HPCForecastScheduler:
    """Schedule jobs at predicted low-cost/low-carbon times."""

    carbon_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)
    carbon_weight: float = 0.5
    cost_weight: float = 0.5
    backend: str = "slurm"

    # --------------------------------------------------
    def submit_at_optimal_time(self, command: Union[str, List[str]], max_delay: float = 21600.0) -> str:
        steps = max(int(max_delay // 3600) + 1, 1)
        carbon_pred = arima_forecast(self.carbon_history, steps=steps)
        cost_pred = arima_forecast(self.cost_history, steps=steps)
        n = min(len(carbon_pred), len(cost_pred))
        delay = 0.0
        if n:
            scores = [
                self.carbon_weight * carbon_pred[i] + self.cost_weight * cost_pred[i]
                for i in range(n)
            ]
            min_idx = int(min(range(n), key=lambda i: scores[i]))
            delay = min_idx * 3600.0
        if delay and delay <= max_delay:
            time.sleep(delay)
        return submit_job(command, backend=self.backend)


__all__ = ["arima_forecast", "HPCForecastScheduler"]
