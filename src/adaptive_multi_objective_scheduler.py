from __future__ import annotations

"""Reinforcement learning scheduler balancing carbon, price, battery and utilization."""

import random
import time
from dataclasses import dataclass, field
from typing import Iterable, Tuple, Dict, Optional, Union, List

try:
    import psutil  # pragma: no cover - optional
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:  # pragma: no cover - optional torch dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    class _DummyCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def memory_allocated() -> int:
            return 0

        @staticmethod
        def get_device_properties(_: int):
            class _P:
                total_memory = 1
            return _P()

    torch = type("torch", (), {"cuda": _DummyCuda})()  # type: ignore

from .telemetry import TelemetryLogger
from .hpc_scheduler import submit_job


@dataclass
class AdaptiveMultiObjectiveScheduler:
    """Q-learning scheduler for multiple resource objectives."""

    history: Iterable[Tuple[float, float, float, float]]
    bins: int = 8
    epsilon: float = 0.1
    alpha: float = 0.5
    gamma: float = 0.9
    check_interval: float = 60.0
    telemetry: Optional[TelemetryLogger] = None
    region: Optional[str] = None
    q: Dict[Tuple[int, int, int, int, int], float] = field(default_factory=dict, init=False)
    min_vals: Tuple[float, float, float, float] = field(default=(0.0, 0.0, 0.0, 0.0), init=False)
    max_vals: Tuple[float, float, float, float] = field(default=(1.0, 1.0, 1.0, 1.0), init=False)

    def __post_init__(self) -> None:
        self.history = list(self.history)
        if self.history:
            cols = list(zip(*self.history))
            self.min_vals = tuple(float(min(c)) for c in cols)
            self.max_vals = tuple(float(max(c)) for c in cols)
            self._train(5)
        self.telemetry = self.telemetry or TelemetryLogger(interval=self.check_interval)

    # --------------------------------------------------
    def _bucket(self, value: float, idx: int) -> int:
        min_v = self.min_vals[idx]
        max_v = self.max_vals[idx]
        if max_v == min_v:
            return 0
        ratio = (value - min_v) / (max_v - min_v)
        return max(0, min(self.bins - 1, int(ratio * (self.bins - 1))))

    # --------------------------------------------------
    def _train(self, cycles: int = 1) -> None:
        for _ in range(cycles):
            for i in range(len(self.history) - 1):
                s_vals = self.history[i]
                sp_vals = self.history[i + 1]
                s = tuple(self._bucket(v, j) for j, v in enumerate(s_vals))
                sp = tuple(self._bucket(v, j) for j, v in enumerate(sp_vals))
                score = sum(s_vals)
                for action, reward in ((0, -score), (1, -0.1)):
                    cur = self.q.get((*s, action), 0.0)
                    next_max = max(self.q.get((*sp, a), 0.0) for a in (0, 1))
                    target = reward + self.gamma * next_max
                    self.q[(*s, action)] = cur + self.alpha * (target - cur)

    # --------------------------------------------------
    def _policy(self, carbon: float, price: float, battery: float, util: float) -> int:
        s = (
            self._bucket(carbon, 0),
            self._bucket(price, 1),
            self._bucket(battery, 2),
            self._bucket(util, 3),
        )
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        run_q = self.q.get((*s, 0), 0.0)
        wait_q = self.q.get((*s, 1), 0.0)
        return 0 if run_q >= wait_q else 1

    # --------------------------------------------------
    def _battery_level(self) -> float:
        if psutil is None:
            return 1.0
        try:
            info = psutil.sensors_battery()
            if info is not None and info.percent is not None:
                return float(info.percent) / 100.0
        except Exception:
            pass
        return 1.0

    # --------------------------------------------------
    def _utilization(self) -> float:
        if torch.cuda.is_available():
            try:
                return (
                    torch.cuda.memory_allocated() /
                    torch.cuda.get_device_properties(0).total_memory
                )
            except Exception:
                return 0.0
        return 0.0

    # --------------------------------------------------
    def submit_job(
        self,
        command: Union[str, List[str]],
        *,
        backend: str = "slurm",
        expected_duration: float = 1.0,
    ) -> str:
        """Submit ``command`` when the learned policy decides to run."""
        start = time.time()
        while True:
            carbon = self.telemetry.get_live_carbon_intensity(self.region)
            price = self.telemetry.get_energy_price(self.region)
            battery = self._battery_level()
            util = self._utilization()
            action = self._policy(carbon, price, battery, util)
            if action == 0:
                job_id = submit_job(command, backend=backend)
                wait = time.time() - start
                self.telemetry.metrics.setdefault("wait_time", 0.0)
                self.telemetry.metrics["wait_time"] += wait
                self.telemetry.metrics.setdefault("energy_usage", 0.0)
                self.telemetry.metrics["energy_usage"] += carbon * expected_duration
                return job_id
            time.sleep(self.check_interval)


__all__ = ["AdaptiveMultiObjectiveScheduler"]
