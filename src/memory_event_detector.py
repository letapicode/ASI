from __future__ import annotations

import json
from typing import Iterable, Dict, Any, List


def detect_change_points(series: Iterable[float], window: int = 5, threshold: float = 2.0) -> List[int]:
    """Return indices where the value deviates ``threshold`` std from the previous ``window``."""
    seq = list(series)
    points = []
    for i in range(window, len(seq)):
        prev = seq[i - window : i]
        mean = sum(prev) / window
        var = sum((x - mean) ** 2 for x in prev) / window
        std = var ** 0.5
        if std > 0:
            if abs(seq[i] - mean) > threshold * std:
                points.append(i)
        elif seq[i] != mean:
            points.append(i)
    return points


class MemoryEventDetector:
    """Detect change points in telemetry metrics."""

    def __init__(self, window: int = 5, threshold: float = 2.0) -> None:
        self.window = window
        self.threshold = threshold
        self.history: List[Dict[str, float]] = []
        self.events: List[Dict[str, Any]] = []

    @staticmethod
    def parse_log(lines: Iterable[str]) -> List[Dict[str, float]]:
        """Parse JSON log lines into metric dictionaries."""
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def update(self, stats: Dict[str, float]) -> List[Dict[str, Any]]:
        """Update history with ``stats`` and record any change-point events."""
        self.history.append(stats)
        events = []
        idx = len(self.history) - 1
        if idx >= self.window:
            for key, val in stats.items():
                series = [h.get(key, 0.0) for h in self.history[idx - self.window : idx]]
                cps = detect_change_points(series + [val], window=self.window, threshold=self.threshold)
                if cps and cps[-1] == self.window:
                    event = {"index": idx, "metric": key, "value": val}
                    self.events.append(event)
                    events.append(event)
        return events


__all__ = ["detect_change_points", "MemoryEventDetector"]
