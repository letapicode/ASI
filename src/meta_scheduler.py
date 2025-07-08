from __future__ import annotations

"""Meta scheduler that selects the best underlying scheduler."""

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Tuple, Union

from .carbon_aware_scheduler import CarbonAwareScheduler
from .rl_carbon_scheduler import RLCarbonScheduler
from .hpc_forecast_scheduler import HPCForecastScheduler

try:  # pragma: no cover - optional
    from .transformer_forecast_scheduler import TransformerForecastScheduler
except Exception:  # pragma: no cover - missing dep
    TransformerForecastScheduler = None  # type: ignore

SchedulerType = Union[
    CarbonAwareScheduler,
    RLCarbonScheduler,
    HPCForecastScheduler,
    "TransformerForecastScheduler",
]


@dataclass
class MetaScheduler:
    """Pick a scheduler based on recent job results."""

    schedulers: Dict[str, SchedulerType] = field(default_factory=dict)
    window: int = 5
    _history: Dict[str, Deque[Tuple[float, float, bool]]] = field(
        default_factory=dict, init=False
    )
    _totals: Dict[str, Tuple[float, float, int]] = field(
        default_factory=dict, init=False
    )

    # --------------------------------------------------
    def record_result(
        self, name: str, success: bool, carbon: float, cost: float
    ) -> None:
        """Store the outcome of a job."""

        hist = self._history.setdefault(name, deque(maxlen=self.window))
        popped = None
        if len(hist) == hist.maxlen:
            popped = hist[0]
        hist.append((carbon, cost, success))

        c_sum, p_sum, fails = self._totals.get(name, (0.0, 0.0, 0))
        c_sum += carbon
        p_sum += cost
        fails += 0 if success else 1
        if popped is not None:
            c_sum -= popped[0]
            p_sum -= popped[1]
            fails -= 0 if popped[2] else 1
        self._totals[name] = (c_sum, p_sum, fails)

    # --------------------------------------------------
    def _score(self, name: str) -> float:
        hist = self._history.get(name)
        if not hist:
            return 0.0
        c_sum, p_sum, fails = self._totals.get(name, (0.0, 0.0, 0))
        n = len(hist)
        avg_c = c_sum / n
        avg_p = p_sum / n
        fail_penalty = fails / n
        return avg_c + avg_p + fail_penalty

    # --------------------------------------------------
    def choose(self) -> Tuple[str, SchedulerType]:
        """Return the name and instance of the best scheduler."""

        best_name = None
        best_score = float("inf")
        for name in self.schedulers:
            score = self._score(name)
            if score < best_score:
                best_score = score
                best_name = name
        if best_name is None:
            raise ValueError("No schedulers configured")
        return best_name, self.schedulers[best_name]

    # --------------------------------------------------
    def _dispatch(
        self, sched: SchedulerType, command: Union[str, List[str]], **kw
    ) -> str:
        if hasattr(sched, "submit_job"):
            return sched.submit_job(command, **kw)  # type: ignore[no-any-return]
        if hasattr(sched, "submit_at_optimal_time"):
            return sched.submit_at_optimal_time(command, **kw)  # type: ignore[no-any-return]
        if hasattr(sched, "schedule_job"):
            _, jid = sched.schedule_job(command, **kw)  # type: ignore[no-any-return]
            return jid
        raise ValueError("Unsupported scheduler")

    # --------------------------------------------------
    def submit_best(
        self, command: Union[str, List[str]], **kw
    ) -> Tuple[str, str]:
        """Submit ``command`` using the best scheduler."""

        name, sched = self.choose()
        jid = self._dispatch(sched, command, **kw)
        return name, jid


__all__ = ["MetaScheduler"]

