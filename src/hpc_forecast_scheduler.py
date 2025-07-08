from __future__ import annotations

"""Schedule HPC jobs based on ARIMA forecasts of cost and carbon intensity."""

from dataclasses import dataclass, field
import time
from typing import List, Union, Tuple, Dict

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

    # cache keyed by (len(carbon_history), len(cost_history), steps)
    _cache: Dict[Tuple[int, int, int], List[float]] = field(default_factory=dict, init=False)

    # --------------------------------------------------
    def forecast_scores(self, max_delay: float, clusters=None) -> List[float]:
        """Return combined carbon/cost forecasts for each hour."""
        steps = max(int(max_delay // 3600) + 1, 1)
        key = (len(self.carbon_history), len(self.cost_history), steps)
        if key in self._cache:
            return list(self._cache[key])
        carbon_pred = arima_forecast(self.carbon_history, steps=steps)
        cost_pred = arima_forecast(self.cost_history, steps=steps)
        n = min(len(carbon_pred), len(cost_pred))
        scores = [
            self.carbon_weight * carbon_pred[i] + self.cost_weight * cost_pred[i]
            for i in range(n)
        ]
        # keep only latest entry to avoid unbounded memory
        self._cache.clear()
        self._cache[key] = list(scores)
        return scores

    # --------------------------------------------------
    def submit_at_optimal_time(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> str:
        scores = self.forecast_scores(max_delay)
        delay = 0.0
        if scores:
            idx = int(min(range(len(scores)), key=lambda i: scores[i]))
            delay = idx * 3600.0
        if delay and delay <= max_delay:
            time.sleep(delay)
        return globals()["submit_job"](command, backend=self.backend)


__all__ = ["arima_forecast", "HPCForecastScheduler"]
