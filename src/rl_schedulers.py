from __future__ import annotations

"""Collection of reinforcement learning schedulers."""

import random
import time
import os
import json
from dataclasses import dataclass, field
from typing import Iterable, Tuple, List, Dict, Union, Optional

from .telemetry import TelemetryLogger
from .hpc_schedulers import submit_job
from .hpc_schedulers import MultiClusterScheduler, _record_carbon_saving
from .forecast_strategies import arima_forecast, _TrendRNN

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - allow running without torch
    torch = None  # type: ignore
    nn = None  # type: ignore


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

    def _bucket(self, intensity: float) -> int:
        if self.max_i == self.min_i:
            return 0
        ratio = (intensity - self.min_i) / (self.max_i - self.min_i)
        return max(0, min(self.bins - 1, int(ratio * (self.bins - 1))))

    def _train(self, cycles: int = 1) -> None:
        for _ in range(cycles):
            for idx in range(len(self.history) - 1):
                i, dur = self.history[idx]
                j, _ = self.history[idx + 1]
                s = self._bucket(i)
                sp = self._bucket(j)
                for action, reward in ((0, -i * dur), (1, -0.1)):
                    cur = self.q.get((s, action), 0.0)
                    next_max = max(self.q.get((sp, a), 0.0) for a in (0, 1))
                    target = reward + self.gamma * next_max
                    self.q[(s, action)] = cur + self.alpha * (target - cur)

    def _policy(self, intensity: float) -> int:
        s = self._bucket(intensity)
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        run_q = self.q.get((s, 0), 0.0)
        wait_q = self.q.get((s, 1), 0.0)
        return 0 if run_q >= wait_q else 1

    def submit_job(
        self,
        command: Union[str, List[str]],
        *,
        backend: str = "slurm",
        expected_duration: float = 1.0,
    ) -> str:
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

    def _bucket(self, value: float, min_v: float, max_v: float) -> int:
        if max_v == min_v:
            return 0
        ratio = (value - min_v) / (max_v - min_v)
        return max(0, min(self.bins - 1, int(ratio * (self.bins - 1))))

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

    def _save(self) -> None:
        if self.qtable_path:
            os.makedirs(os.path.dirname(self.qtable_path) or ".", exist_ok=True)
            data = {
                "q1": {f"{c},{p},{a}": v for (c, p, a), v in self.q1.items()},
                "q2": {f"{c},{p},{a}": v for (c, p, a), v in self.q2.items()},
            }
            with open(self.qtable_path, "w") as fh:
                json.dump(data, fh)

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

    def submit_best(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> Tuple[str, str]:
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
                    sched.carbon_weight * carbon_pred[i] + sched.cost_weight * price_pred[i]
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


@dataclass
class CoordinatedRLCostScheduler(RLCostScheduler):
    """Share Q-values among peer schedulers using a lightweight group."""

    peers: Dict[str, RLCostScheduler] = field(default_factory=dict)
    group: "_Group" = field(default_factory=lambda: _Group(), init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.group.add(self)

    def register_peer(self, name: str, sched: RLCostScheduler) -> None:
        if sched is self:
            return
        self.peers[name] = sched
        if isinstance(sched, CoordinatedRLCostScheduler):
            if self.group is not sched.group:
                self.group.merge(sched.group)
            self.group.add(sched)
            sched.group = self.group

    def exchange_values(self) -> None:
        self.group.sync()

    def _peer_q(self, state: Tuple[int, int], action: int) -> float:
        key = (state[0], state[1], action)
        members = self.group.members or [self]
        total = 0.0
        for m in members:
            total += m.q1.get(key, 0.0) + m.q2.get(key, 0.0)
        return total / len(members)

    def _policy(self, carbon: float, price: float) -> int:
        self.exchange_values()
        s = (
            self._bucket(carbon, self.min_c, self.max_c),
            self._bucket(price, self.min_p, self.max_p),
        )
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        run_q = self._peer_q(s, 0)
        wait_q = self._peer_q(s, 1)
        return 0 if run_q >= wait_q else 1

    def submit_best(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> Tuple[str, str]:
        return super().submit_best(command, max_delay)


@dataclass
class _Group:
    members: List[CoordinatedRLCostScheduler] = field(default_factory=list)

    def add(self, sched: CoordinatedRLCostScheduler) -> None:
        if sched not in self.members:
            self.members.append(sched)

    def merge(self, other: "_Group") -> None:
        for m in other.members:
            self.add(m)
        for m in self.members:
            m.group = self

    def sync(self) -> None:
        q1_sum: Dict[Tuple[int, int, int], float] = {}
        q1_cnt: Dict[Tuple[int, int, int], int] = {}
        q2_sum: Dict[Tuple[int, int, int], float] = {}
        q2_cnt: Dict[Tuple[int, int, int], int] = {}
        for s in self.members:
            for k, v in s.q1.items():
                q1_sum[k] = q1_sum.get(k, 0.0) + v
                q1_cnt[k] = q1_cnt.get(k, 0) + 1
            for k, v in s.q2.items():
                q2_sum[k] = q2_sum.get(k, 0.0) + v
                q2_cnt[k] = q2_cnt.get(k, 0) + 1
        for k in q1_sum:
            avg = q1_sum[k] / q1_cnt[k]
            for s in self.members:
                s.q1[k] = avg
        for k in q2_sum:
            avg = q2_sum[k] / q2_cnt[k]
            for s in self.members:
                s.q2[k] = avg


@dataclass
class RLMultiClusterScheduler(MultiClusterScheduler):
    """Choose HPC clusters with a simple Q-learning policy."""

    alpha: float = 0.5
    epsilon: float = 0.1
    q_table: Dict[Tuple[str, int], float] = field(default_factory=dict)
    last_queue: Dict[str, float] = field(default_factory=dict)

    def update_policy(self, log_entry: Dict[str, float | str]) -> None:
        cluster = str(log_entry.get("cluster"))
        hour = int(log_entry.get("hour", 0)) % 24
        queue = float(log_entry.get("queue_time", 0.0))
        duration = float(log_entry.get("duration", 0.0))
        carbon = float(log_entry.get("carbon", 0.0))
        reward = -(queue + duration) - carbon * duration
        key = (cluster, hour)
        old = self.q_table.get(key, 0.0)
        self.q_table[key] = old + self.alpha * (reward - old)
        self.last_queue[cluster] = queue

    def submit_best_rl(
        self,
        command: Union[str, List[str]],
        max_delay: float = 21600.0,
        *,
        expected_duration: float = 1.0,
    ) -> Tuple[str, str]:
        hour = int(time.time() // 3600) % 24
        clusters = list(self.clusters.keys())
        if random.random() < self.epsilon or not self.q_table:
            choice = random.choice(clusters)
            backend = self.clusters[choice].backend
            tel = self.telemetry.get(choice) if self.telemetry else None
            job_id = submit_job(command, backend=backend, telemetry=tel)
            _record_carbon_saving(
                self.telemetry,
                tel,
                choice,
                expected_duration,
                self.schedule_log,
                self.dashboard,
            )
            return choice, job_id
        best_cluster = clusters[0]
        best_val = self.q_table.get((best_cluster, hour), float("-inf"))
        for name in clusters[1:]:
            val = self.q_table.get((name, hour), float("-inf"))
            if val > best_val:
                best_val = val
                best_cluster = name
        backend = self.clusters[best_cluster].backend
        tel = self.telemetry.get(best_cluster) if self.telemetry else None
        job_id = submit_job(command, backend=backend, telemetry=tel)
        _record_carbon_saving(
            self.telemetry,
            tel,
            best_cluster,
            expected_duration,
            self.schedule_log,
            self.dashboard,
        )
        return best_cluster, job_id


@dataclass
class DeepRLScheduler(MultiClusterScheduler):
    """Predict cost and carbon trends with an RNN to choose the best cluster."""

    hidden_size: int = 16
    num_layers: int = 2
    history_len: int = 4
    lr: float = 0.01
    epochs: int = 20
    model: object | None = field(default=None, init=False)

    def update_history(self, cluster: str, carbon: float, cost: float) -> None:
        sched = self.clusters.get(cluster)
        if sched is None:
            raise KeyError(cluster)
        sched.carbon_history.append(carbon)
        sched.cost_history.append(cost)

    def refit(self) -> None:
        if self.model is None:
            return
        self.fit()

    def __post_init__(self) -> None:
        if torch is None:
            self.model = None
            return
        self.model = _TrendRNN(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        self.fit()

    def fit(self) -> None:  # pragma: no cover - tiny dataset
        if torch is None:
            return
        dataset: List[Tuple[List[List[float]], List[float]]] = []
        for sched in self.clusters.values():
            ch, ph = sched.carbon_history, sched.cost_history
            n = min(len(ch), len(ph))
            for i in range(self.history_len, n):
                x = [[ch[j], ph[j]] for j in range(i - self.history_len, i)]
                y = [ch[i], ph[i]]
                dataset.append((x, y))
        if not dataset:
            return
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            for x, y in dataset:
                x_t = torch.tensor([x], dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32)
                optim.zero_grad()
                pred = self.model(x_t)
                loss = loss_fn(pred, y_t)
                loss.backward()
                optim.step()

    def _predict(self, sched, steps: int) -> Tuple[List[float], List[float]]:
        ch, ph = sched.carbon_history, sched.cost_history
        n = min(len(ch), len(ph))
        if torch is None or self.model is None or n < self.history_len:
            last_c = ch[-1] if ch else 0.0
            last_p = ph[-1] if ph else 0.0
            return [last_c] * steps, [last_p] * steps
        seq = [[ch[i], ph[i]] for i in range(n - self.history_len, n)]
        cur = torch.tensor([seq], dtype=torch.float32)
        preds_c, preds_p = [], []
        with torch.no_grad():
            for _ in range(steps):
                out = self.model(cur)
                c, p = out[0].tolist()
                preds_c.append(float(c))
                preds_p.append(float(p))
                nxt = torch.tensor([[c, p]], dtype=torch.float32).unsqueeze(0)
                cur = torch.cat([cur[:, 1:], nxt], dim=1)
        return preds_c, preds_p

    def schedule_job(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> Tuple[str, str]:
        best_cluster = None
        best_backend = None
        best_score = float("inf")
        best_delay = 0.0
        steps = max(int(max_delay // 3600) + 1, 1)
        for name, sched in self.clusters.items():
            c_pred, p_pred = self._predict(sched, steps)
            n = min(len(c_pred), len(p_pred))
            if not n:
                continue
            scores = [
                sched.carbon_weight * c_pred[i] + sched.cost_weight * p_pred[i]
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
        if best_delay and best_delay <= max_delay:
            time.sleep(best_delay)
        job_id = submit_job(command, backend=best_backend)
        return best_cluster, job_id


__all__ = [
    "RLCarbonScheduler",
    "RLCostScheduler",
    "CoordinatedRLCostScheduler",
    "RLMultiClusterScheduler",
    "DeepRLScheduler",
]
