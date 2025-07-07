"""Background updater for retrieval RL policy."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Tuple

from .retrieval_rl import RetrievalPolicy, train_policy
from .telemetry import TelemetryLogger


@dataclass
class RetrievalPolicyUpdater:
    """Periodically train ``RetrievalPolicy`` from query logs."""

    policy: RetrievalPolicy
    log_source: Callable[[], Iterable[Tuple[Any, bool, float]]]
    interval: float = 60.0
    telemetry: TelemetryLogger | None = None
    baseline_recall: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None
        if self.telemetry is None:
            self.telemetry = TelemetryLogger(interval=self.interval)

    # --------------------------------------------------
    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # --------------------------------------------------
    def stop(self) -> None:
        if self.thread is None:
            return
        self._stop.set()
        self.thread.join(timeout=1.0)
        self.thread = None
        self._stop.clear()

    # --------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._update()
            self._stop.wait(self.interval)

    # --------------------------------------------------
    def _update(self) -> None:
        try:
            entries = list(self.log_source())
        except Exception:
            return
        if not entries:
            return
        rewards = []
        hits = 0
        for meta, hit, latency in entries:
            reward = (1.0 if hit else 0.0) - float(latency)
            rewards.append((meta, reward))
            if hit:
                hits += 1
        recall = hits / len(entries)
        improvement = recall - self.baseline_recall
        self.baseline_recall += 0.1 * improvement
        train_policy(self.policy, rewards)
        if self.telemetry is not None:
            self.telemetry.events.append(
                {
                    "index": len(self.telemetry.events),
                    "metric": "recall_improvement",
                    "value": improvement,
                }
            )


__all__ = ["RetrievalPolicyUpdater"]
