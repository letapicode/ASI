from __future__ import annotations

"""Schedule HPC jobs based on ARIMA forecasts of cost and carbon intensity."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .hpc_base_scheduler import HPCBaseScheduler, ForecastStrategy


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


class ArimaStrategy(ForecastStrategy):
    """ARIMA-based forecasting helper."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, int], List[float]] = {}

    def forecast_scores(
        self,
        scheduler: HPCBaseScheduler,
        max_delay: float,
        clusters=None,
    ) -> List[float]:
        steps = max(int(max_delay // 3600) + 1, 1)
        key = (len(scheduler.carbon_history), len(scheduler.cost_history), steps)
        if key in self._cache:
            return list(self._cache[key])
        carbon_pred = arima_forecast(scheduler.carbon_history, steps=steps)
        cost_pred = arima_forecast(scheduler.cost_history, steps=steps)
        n = min(len(carbon_pred), len(cost_pred))
        scores = [
            scheduler.carbon_weight * carbon_pred[i]
            + scheduler.cost_weight * cost_pred[i]
            for i in range(n)
        ]
        self._cache.clear()
        self._cache[key] = list(scores)
        return scores


@dataclass
class HPCForecastScheduler(HPCBaseScheduler):
    """Schedule jobs at predicted low-cost/low-carbon times using ARIMA."""

    def __post_init__(self) -> None:  # pragma: no cover - simple init
        self.strategy = ArimaStrategy()


__all__ = ["arima_forecast", "HPCForecastScheduler", "ArimaStrategy"]
