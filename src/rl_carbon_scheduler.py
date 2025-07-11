from __future__ import annotations

"""Reinforcement learning based carbon-aware scheduler."""

import time
from typing import Iterable, Tuple, List, Dict, Union, Optional

from .telemetry import TelemetryLogger
from .hpc_schedulers import submit_job
from .rl_scheduler_base import RLSchedulerBase


class RLCarbonScheduler(RLSchedulerBase):
    """Schedule jobs using a Q-learning policy trained on historical data."""

    def __init__(
        self,
        history: Iterable[Tuple[float, float]],
        *,
        bins: int = 10,
        epsilon: float = 0.1,
        alpha: float = 0.5,
        gamma: float = 0.9,
        check_interval: float = 60.0,
        telemetry: Optional[TelemetryLogger] = None,
        region: Optional[str] = None,
    ) -> None:
        super().__init__(bins=bins, epsilon=epsilon, alpha=alpha, gamma=gamma)
        self.history = list(history)
        self.check_interval = check_interval
        self.telemetry = telemetry or TelemetryLogger(interval=check_interval)
        self.region = region
        if self.history:
            self.min_i = min(i for i, _ in self.history)
            self.max_i = max(i for i, _ in self.history)
        else:
            self.min_i = 0.0
            self.max_i = 1.0
        self.configure_state_bounds([self.min_i], [self.max_i])
        self.q: Dict[Tuple[int, int], float] = self.q1  # backward compatibility
        if self.history:
            traces = [
                (
                    (self.history[i][0],),
                    (self.history[i + 1][0],),
                    -self.history[i][0] * self.history[i][1],
                )
                for i in range(len(self.history) - 1)
            ]
            super()._train(traces, cycles=10)

    # --------------------------------------------------------------
    def submit_job(
        self,
        command: Union[str, List[str]],
        *,
        backend: str = "slurm",
        expected_duration: float = 1.0,
    ) -> str:
        """Submit ``command`` when the RL policy decides to run."""
        start = time.time()
        while True:
            intensity = self.telemetry.get_carbon_intensity(self.region)
            action = self._policy(intensity)
            if action == 0:
                job_id = submit_job(
                    command,
                    backend=backend,
                    telemetry=self.telemetry,
                    region=self.region,
                )
                wait = time.time() - start
                energy = intensity * expected_duration
                self.telemetry.metrics["energy_usage"] = (
                    self.telemetry.metrics.get("energy_usage", 0.0) + energy
                )
                self.telemetry.metrics["wait_time"] = (
                    self.telemetry.metrics.get("wait_time", 0.0) + wait
                )
                return job_id
            time.sleep(self.check_interval)


__all__ = ["RLCarbonScheduler"]

