from __future__ import annotations

"""Common Q-learning helpers for RL-based schedulers."""

import random
from typing import Dict, Iterable, List, Sequence, Tuple, Optional


class RLSchedulerBase:
    """Utility base class implementing basic Q-learning."""

    def __init__(
        self,
        *,
        bins: int = 10,
        epsilon: float = 0.1,
        alpha: float = 0.5,
        gamma: float = 0.9,
        double_q: bool = False,
    ) -> None:
        self.bins = bins
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.double_q = double_q
        self.q1: Dict[Tuple[int, ...], float] = {}
        self.q2: Optional[Dict[Tuple[int, ...], float]] = {} if double_q else None
        self.state_mins: List[float] = []
        self.state_maxs: List[float] = []

    # --------------------------------------------------
    def configure_state_bounds(self, mins: Sequence[float], maxs: Sequence[float]) -> None:
        """Set min and max values for each feature dimension."""
        self.state_mins = list(mins)
        self.state_maxs = list(maxs)

    # --------------------------------------------------
    def _bucket(self, value: float, min_v: float, max_v: float) -> int:
        if max_v == min_v:
            return 0
        ratio = (value - min_v) / (max_v - min_v)
        return max(0, min(self.bins - 1, int(ratio * (self.bins - 1))))

    # --------------------------------------------------
    def _state_key(self, values: Sequence[float]) -> Tuple[int, ...]:
        return tuple(
            self._bucket(v, mn, mx)
            for v, mn, mx in zip(values, self.state_mins, self.state_maxs)
        )

    # --------------------------------------------------
    def _train(
        self,
        traces: Iterable[Tuple[Sequence[float], Sequence[float], float]],
        cycles: int = 1,
    ) -> None:
        """Update Q-tables using provided ``traces``."""
        traces = list(traces)
        if not traces:
            return
        for _ in range(cycles):
            for state, next_state, run_reward in traces:
                s = self._state_key(state)
                sp = self._state_key(next_state)
                for action, reward in ((0, run_reward), (1, -0.1)):
                    if self.double_q and self.q2 is not None:
                        if random.random() < 0.5:
                            q_cur = self.q1
                            q_other = self.q2
                        else:
                            q_cur = self.q2
                            q_other = self.q1
                        cur = q_cur.get((*s, action), 0.0)
                        best_a = 0
                        best_val = -float("inf")
                        for a in (0, 1):
                            val = q_cur.get((*sp, a), 0.0) + q_other.get((*sp, a), 0.0)
                            if val > best_val:
                                best_val = val
                                best_a = a
                        next_q = q_other.get((*sp, best_a), 0.0)
                        target = reward + self.gamma * next_q
                        q_cur[(*s, action)] = cur + self.alpha * (target - cur)
                    else:
                        cur = self.q1.get((*s, action), 0.0)
                        next_max = max(self.q1.get((*sp, a), 0.0) for a in (0, 1))
                        target = reward + self.gamma * next_max
                        self.q1[(*s, action)] = cur + self.alpha * (target - cur)

    # --------------------------------------------------
    def _policy(self, *state: float) -> int:
        s = self._state_key(state)
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        if self.double_q and self.q2 is not None:
            run_q = self.q1.get((*s, 0), 0.0) + self.q2.get((*s, 0), 0.0)
            wait_q = self.q1.get((*s, 1), 0.0) + self.q2.get((*s, 1), 0.0)
        else:
            run_q = self.q1.get((*s, 0), 0.0)
            wait_q = self.q1.get((*s, 1), 0.0)
        return 0 if run_q >= wait_q else 1


__all__ = ["RLSchedulerBase"]
