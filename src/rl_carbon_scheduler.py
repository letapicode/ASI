from __future__ import annotations

"""Reinforcement learning based carbon-aware scheduler."""

import random
import time
from typing import Iterable, Tuple, List, Dict, Union, Optional

from .telemetry import TelemetryLogger
from .hpc_schedulers import submit_job


class RLCarbonScheduler:
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
        self.history = list(history)
        self.bins = bins
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.check_interval = check_interval
        self.telemetry = telemetry or TelemetryLogger(interval=check_interval)
        self.region = region
        if self.history:
            self.min_i = min(i for i, _ in self.history)
            self.max_i = max(i for i, _ in self.history)
        else:
            self.min_i = 0.0
            self.max_i = 1.0
        self.q: Dict[Tuple[int, int], float] = {}
        if self.history:
            self._train(10)

    # --------------------------------------------------------------
    def _bucket(self, intensity: float) -> int:
        if self.max_i == self.min_i:
            return 0
        ratio = (intensity - self.min_i) / (self.max_i - self.min_i)
        return max(0, min(self.bins - 1, int(ratio * (self.bins - 1))))

    # --------------------------------------------------------------
    def _train(self, cycles: int = 1) -> None:
        for _ in range(cycles):
            for idx in range(len(self.history) - 1):
                i, dur = self.history[idx]
                j, _ = self.history[idx + 1]
                s = self._bucket(i)
                sp = self._bucket(j)
                for action, reward in ((0, -i * dur), (1, -0.1)):
                    cur = self.q.get((s, action), 0.0)
                    next_max = max(
                        self.q.get((sp, a), 0.0) for a in (0, 1)
                    )
                    target = reward + self.gamma * next_max
                    self.q[(s, action)] = cur + self.alpha * (target - cur)

    # --------------------------------------------------------------
    def _policy(self, intensity: float) -> int:
        s = self._bucket(intensity)
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        run_q = self.q.get((s, 0), 0.0)
        wait_q = self.q.get((s, 1), 0.0)
        return 0 if run_q >= wait_q else 1

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

