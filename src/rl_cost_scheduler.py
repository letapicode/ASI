from __future__ import annotations

"""Reinforcement learning scheduler balancing carbon intensity and price."""

from dataclasses import dataclass, field
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

from .hpc_multi_scheduler import MultiClusterScheduler
from .hpc_forecast_scheduler import arima_forecast
from .hpc_schedulers import submit_job
from .rl_scheduler_base import RLSchedulerBase


@dataclass
class RLCostScheduler(MultiClusterScheduler, RLSchedulerBase):
    """Learn when to submit jobs using carbon and cost traces."""

    bins: int = 10
    epsilon: float = 0.1
    alpha: float = 0.5
    gamma: float = 0.9
    check_interval: float = 60.0
    qtable_path: Optional[str] = None
    q1: Dict[Tuple[int, int, int], float] = field(default_factory=dict, init=False)
    q2: Dict[Tuple[int, int, int], float] = field(default_factory=dict, init=False)
    min_c: float = field(default=0.0, init=False)
    max_c: float = field(default=1.0, init=False)
    min_p: float = field(default=0.0, init=False)
    max_p: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        RLSchedulerBase.__init__(
            self,
            bins=self.bins,
            epsilon=self.epsilon,
            alpha=self.alpha,
            gamma=self.gamma,
            double_q=True,
        )
        if self.qtable_path and os.path.exists(self.qtable_path):
            with open(self.qtable_path) as fh:
                data = json.load(fh)
                self.q1 = {
                    tuple(map(int, k.split(','))): float(v)
                    for k, v in data.get('q1', {}).items()
                }
                self.q2 = {
                    tuple(map(int, k.split(','))): float(v)
                    for k, v in data.get('q2', {}).items()
                }
        c_vals: List[float] = []
        p_vals: List[float] = []
        for sched in self.clusters.values():
            c_vals.extend(list(sched.carbon_history))
            p_vals.extend(list(sched.cost_history))
        if c_vals:
            self.min_c = min(c_vals)
            self.max_c = max(c_vals)
        if p_vals:
            self.min_p = min(p_vals)
            self.max_p = max(p_vals)
        self.configure_state_bounds([self.min_c, self.min_p], [self.max_c, self.max_p])
        if (c_vals or p_vals) and not (self.q1 or self.q2):
            self._train(10)

    # --------------------------------------------------
    def _train(self, cycles: int = 1) -> None:
        traces: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
        for sched in self.clusters.values():
            n = min(len(sched.carbon_history), len(sched.cost_history))
            for i in range(n - 1):
                traces.append(
                    (
                        (sched.carbon_history[i], sched.cost_history[i]),
                        (sched.carbon_history[i + 1], sched.cost_history[i + 1]),
                        -(sched.carbon_history[i] + sched.cost_history[i]),
                    )
                )
        super()._train(traces, cycles)
        if traces:
            self.epsilon = max(self.epsilon * 0.99, 0.01)
            self._save()

    # --------------------------------------------------
    def _save(self) -> None:
        if self.qtable_path:
            os.makedirs(os.path.dirname(self.qtable_path) or ".", exist_ok=True)
            data = {
                "q1": {f"{c},{p},{a}": v for (c, p, a), v in self.q1.items()},
                "q2": {f"{c},{p},{a}": v for (c, p, a), v in self.q2.items()},
            }
            with open(self.qtable_path, "w") as fh:
                json.dump(data, fh)


    # --------------------------------------------------
    def submit_best(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> Tuple[str, str]:
        """Return chosen cluster name and job id using the RL policy."""
        self._train(1)
        while True:
            best_cluster = None
            best_backend = None
            best_score = float("inf")
            best_delay = 0.0
            best_c = 0.0
            best_p = 0.0

            for name, sched in self.clusters.items():
                steps = max(int(max_delay // 3600) + 1, 1)
                carbon_pred = arima_forecast(sched.carbon_history, steps=steps)
                price_pred = arima_forecast(sched.cost_history, steps=steps)
                n = min(len(carbon_pred), len(price_pred))
                if not n:
                    continue
                scores = [
                    sched.carbon_weight * carbon_pred[i]
                    + sched.cost_weight * price_pred[i]
                    for i in range(n)
                ]
                idx = int(min(range(n), key=lambda i: scores[i]))
                if scores[idx] < best_score:
                    best_score = scores[idx]
                    best_delay = idx * 3600.0
                    best_cluster = name
                    best_backend = sched.backend
                    best_c = carbon_pred[idx]
                    best_p = price_pred[idx]

            if best_cluster is None:
                raise ValueError("No forecasts available to choose a cluster")

            action = self._policy(best_c, best_p)
            if action == 0 or best_delay > max_delay:
                if best_delay and best_delay <= max_delay:
                    time.sleep(best_delay)
                job_id = submit_job(command, backend=best_backend)
                return best_cluster, job_id
            time.sleep(self.check_interval)


__all__ = ["RLCostScheduler"]
