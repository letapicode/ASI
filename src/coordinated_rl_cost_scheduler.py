from __future__ import annotations

"""Cooperative extension of :class:`RLCostScheduler`."""

import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Union

from .rl_cost_scheduler import RLCostScheduler


@dataclass
class _Group:
    members: List["CoordinatedRLCostScheduler"] = field(default_factory=list)

    def add(self, sched: "CoordinatedRLCostScheduler") -> None:
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
class CoordinatedRLCostScheduler(RLCostScheduler):
    """Share Q-values among peer schedulers using a lightweight group."""

    peers: Dict[str, RLCostScheduler] = field(default_factory=dict)
    group: _Group = field(default_factory=_Group, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.group.add(self)

    # --------------------------------------------------
    def register_peer(self, name: str, sched: RLCostScheduler) -> None:
        """Add ``sched`` as a coordination peer."""
        if sched is self:
            return
        self.peers[name] = sched
        if isinstance(sched, CoordinatedRLCostScheduler):
            if self.group is not sched.group:
                self.group.merge(sched.group)
            self.group.add(sched)
            sched.group = self.group

    # --------------------------------------------------
    def exchange_values(self) -> None:
        """Average Q-values across the whole group."""
        self.group.sync()

    # --------------------------------------------------
    def _peer_q(self, state: Tuple[int, int], action: int) -> float:
        key = (state[0], state[1], action)
        members = self.group.members or [self]
        total = 0.0
        for m in members:
            total += m.q1.get(key, 0.0) + m.q2.get(key, 0.0)
        return total / len(members)

    # --------------------------------------------------
    def _policy(self, carbon: float, price: float) -> int:
        self.exchange_values()
        s = self._state_key((carbon, price))
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        run_q = self._peer_q(s, 0)
        wait_q = self._peer_q(s, 1)
        return 0 if run_q >= wait_q else 1

    # --------------------------------------------------
    def submit_best(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> Tuple[str, str]:
        return super().submit_best(command, max_delay)


__all__ = ["CoordinatedRLCostScheduler"]
