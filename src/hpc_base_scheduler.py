from __future__ import annotations

"""Base utilities for HPC job schedulers with pluggable forecasting."""

from dataclasses import dataclass, field
import time
from typing import List, Protocol, Union, Dict

from .hpc_schedulers import submit_job


class ForecastStrategy(Protocol):
    """Interface for forecasting cluster cost/carbon scores."""

    def forecast_scores(
        self,
        scheduler: "HPCBaseScheduler",
        max_delay: float,
        clusters: Dict[str, "HPCBaseScheduler"] | None = None,
    ) -> List[float]:
        ...


@dataclass
class HPCBaseScheduler:
    """Manage job queueing and submission using a forecasting strategy."""

    carbon_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)
    carbon_weight: float = 0.5
    cost_weight: float = 0.5
    backend: str = "slurm"
    strategy: ForecastStrategy | None = field(default=None, repr=False)

    _queue: List[Union[str, List[str]]] = field(default_factory=list, init=False, repr=False)

    # --------------------------------------------------
    def forecast_scores(
        self, max_delay: float, clusters: Dict[str, "HPCBaseScheduler"] | None = None
    ) -> List[float]:
        """Delegate forecasting to the attached strategy."""
        if self.strategy is None:
            raise NotImplementedError("No forecasting strategy configured")
        return self.strategy.forecast_scores(self, max_delay, clusters)

    # --------------------------------------------------
    def submit_at_optimal_time(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> str:
        """Submit ``command`` when the forecast is most favourable."""
        scores = self.forecast_scores(max_delay)
        delay = 0.0
        if scores:
            idx = int(min(range(len(scores)), key=lambda i: scores[i]))
            delay = idx * 3600.0
        if delay and delay <= max_delay:
            time.sleep(delay)
        return submit_job(command, backend=self.backend)

    # --------------------------------------------------
    def queue_job(self, command: Union[str, List[str]]) -> None:
        """Add a command to the local queue."""
        self._queue.append(command)

    # --------------------------------------------------
    def run_queue(self, max_delay: float = 21600.0) -> List[str]:
        """Submit all queued commands sequentially."""
        results: List[str] = []
        while self._queue:
            cmd = self._queue.pop(0)
            results.append(self.submit_at_optimal_time(cmd, max_delay))
        return results


__all__ = ["HPCBaseScheduler", "ForecastStrategy"]
