from __future__ import annotations

"""Reinforcement learning scheduler balancing carbon intensity and price."""

from dataclasses import dataclass, field
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

from .hpc_multi_scheduler import MultiClusterScheduler
from .hpc_forecast_scheduler import arima_forecast
from .hpc_schedulers import submit_job


@dataclass
class RLCostScheduler(MultiClusterScheduler):
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
        if (c_vals or p_vals) and not (self.q1 or self.q2):
            self._train(10)

    # --------------------------------------------------
    def _bucket(self, value: float, min_v: float, max_v: float) -> int:
        if max_v == min_v:
            return 0
        ratio = (value - min_v) / (max_v - min_v)
        return max(0, min(self.bins - 1, int(ratio * (self.bins - 1))))

    # --------------------------------------------------
    def _train(self, cycles: int = 1) -> None:
        traces: List[Tuple[float, float]] = []
        for sched in self.clusters.values():
            n = min(len(sched.carbon_history), len(sched.cost_history))
            for i in range(n):
                traces.append((sched.carbon_history[i], sched.cost_history[i]))
        for _ in range(cycles):
            for i in range(len(traces) - 1):
                c, p = traces[i]
                c_next, p_next = traces[i + 1]
                s = (
                    self._bucket(c, self.min_c, self.max_c),
                    self._bucket(p, self.min_p, self.max_p),
                )
                sp = (
                    self._bucket(c_next, self.min_c, self.max_c),
                    self._bucket(p_next, self.min_p, self.max_p),
                )
                score = c + p
                for action, reward in ((0, -score), (1, -0.1)):
                    if random.random() < 0.5:
                        q_cur = self.q1
                        q_other = self.q2
                    else:
                        q_cur = self.q2
                        q_other = self.q1
                    cur = q_cur.get((s[0], s[1], action), 0.0)
                    best_a = 0
                    best_val = -float("inf")
                    for a in (0, 1):
                        val = q_cur.get((sp[0], sp[1], a), 0.0) + q_other.get((sp[0], sp[1], a), 0.0)
                        if val > best_val:
                            best_val = val
                            best_a = a
                    next_q = q_other.get((sp[0], sp[1], best_a), 0.0)
                    target = reward + self.gamma * next_q
                    q_cur[(s[0], s[1], action)] = cur + self.alpha * (target - cur)
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
    def _policy(self, carbon: float, price: float) -> int:
        s = (
            self._bucket(carbon, self.min_c, self.max_c),
            self._bucket(price, self.min_p, self.max_p),
        )
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        run_q = self.q1.get((s[0], s[1], 0), 0.0) + self.q2.get((s[0], s[1], 0), 0.0)
        wait_q = self.q1.get((s[0], s[1], 1), 0.0) + self.q2.get((s[0], s[1], 1), 0.0)
        return 0 if run_q >= wait_q else 1

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
