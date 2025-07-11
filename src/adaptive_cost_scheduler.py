from __future__ import annotations

"""Adaptive scheduler that learns from carbon and price histories."""

from dataclasses import dataclass, field
import json
import os
import random
import time
from typing import Dict, List, Optional, Union, Tuple

from .hpc_multi_scheduler import MultiClusterScheduler
from .forecast_strategies import arima_forecast
from .hpc_schedulers import submit_job


@dataclass
class AdaptiveCostScheduler(MultiClusterScheduler):
    """Learn when to submit jobs using a simple Q-learning policy."""

    bins: int = 10
    epsilon: float = 0.1
    alpha: float = 0.5
    gamma: float = 0.9
    check_interval: float = 60.0
    qtable_path: Optional[str] = None
    q: Dict[Tuple[int, int], float] = field(default_factory=dict, init=False)
    min_score: float = field(default=0.0, init=False)
    max_score: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        if self.qtable_path and os.path.exists(self.qtable_path):
            with open(self.qtable_path) as fh:
                data = json.load(fh)
                self.q = {
                    (int(k.split(',')[0]), int(k.split(',')[1])): float(v)
                    for k, v in data.items()
                }
        scores = []
        for sched in self.clusters.values():
            for c, p in zip(sched.carbon_history, sched.cost_history):
                scores.append(sched.carbon_weight * c + sched.cost_weight * p)
        if scores:
            self.min_score = min(scores)
            self.max_score = max(scores)
            if not self.q:
                self._train(10)

    # --------------------------------------------------
    def _bucket(self, score: float) -> int:
        if self.max_score == self.min_score:
            return 0
        ratio = (score - self.min_score) / (self.max_score - self.min_score)
        return max(0, min(self.bins - 1, int(ratio * (self.bins - 1))))

    # --------------------------------------------------
    def _train(self, cycles: int = 1) -> None:
        hist: List[float] = []
        for sched in self.clusters.values():
            for c, p in zip(sched.carbon_history, sched.cost_history):
                hist.append(sched.carbon_weight * c + sched.cost_weight * p)
        for _ in range(cycles):
            for idx in range(len(hist) - 1):
                s = self._bucket(hist[idx])
                sp = self._bucket(hist[idx + 1])
                for action, reward in ((0, -hist[idx]), (1, -0.1)):
                    cur = self.q.get((s, action), 0.0)
                    next_max = max(self.q.get((sp, a), 0.0) for a in (0, 1))
                    target = reward + self.gamma * next_max
                    self.q[(s, action)] = cur + self.alpha * (target - cur)
        self._save()

    # --------------------------------------------------
    def _save(self) -> None:
        if self.qtable_path:
            os.makedirs(os.path.dirname(self.qtable_path) or ".", exist_ok=True)
            data = {f"{s},{a}": v for (s, a), v in self.q.items()}
            with open(self.qtable_path, "w") as fh:
                json.dump(data, fh)

    # --------------------------------------------------
    def _policy(self, score: float) -> int:
        s = self._bucket(score)
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        run_q = self.q.get((s, 0), 0.0)
        wait_q = self.q.get((s, 1), 0.0)
        return 0 if run_q >= wait_q else 1

    # --------------------------------------------------
    def submit_best(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> Tuple[str, str]:
        # refresh policy with the latest history
        self._train(1)
        while True:
            best_cluster = None
            best_backend = None
            best_score = float("inf")
            best_delay = 0.0

            for name, sched in self.clusters.items():
                steps = max(int(max_delay // 3600) + 1, 1)
                carbon_pred = arima_forecast(sched.carbon_history, steps=steps)
                cost_pred = arima_forecast(sched.cost_history, steps=steps)
                n = min(len(carbon_pred), len(cost_pred))
                if not n:
                    continue
                scores = [
                    sched.carbon_weight * carbon_pred[i]
                    + sched.cost_weight * cost_pred[i]
                    for i in range(n)
                ]
                idx = int(min(range(n), key=lambda i: scores[i]))
                if scores[idx] < best_score:
                    best_score = scores[idx]
                    best_delay = idx * 3600.0
                    best_cluster = name
                    best_backend = sched.backend

            if best_cluster is None:
                raise ValueError("No forecasts available to choose a cluster")

            action = self._policy(best_score)
            if action == 0 or best_delay > max_delay:
                if best_delay and best_delay <= max_delay:
                    time.sleep(best_delay)
                job_id = submit_job(command, backend=best_backend)
                return best_cluster, job_id
            time.sleep(self.check_interval)


__all__ = ["AdaptiveCostScheduler"]
