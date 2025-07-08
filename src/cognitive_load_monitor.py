from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Iterable

try:
    from prometheus_client import Gauge  # pragma: no cover - optional
    _HAS_PROM = True
except Exception:  # pragma: no cover - optional
    Gauge = None
    _HAS_PROM = False

from .telemetry import TelemetryLogger


@dataclass
class CognitiveLoadMonitor:
    """Track user pause durations and correction rates."""

    telemetry: TelemetryLogger | None = None
    pause_threshold: float = 2.0
    pauses: List[float] = field(default_factory=list)
    corrections: int = 0
    total_inputs: int = 0
    callbacks: List[Callable[[float], None]] = field(default_factory=list)
    _last_time: float | None = None

    def __post_init__(self) -> None:
        self.telemetry = self.telemetry or TelemetryLogger(interval=0.5)
        if _HAS_PROM:
            for name in ["avg_pause", "correction_rate", "cognitive_load"]:
                self.telemetry.metrics.setdefault(name, Gauge(name, name))

    def add_callback(self, callback: Callable[[float], None]) -> None:
        """Register a callback to receive load updates."""
        self.callbacks.append(callback)

    # --------------------------------------------------------------
    def log_input(self, text: str = "", timestamp: Optional[float] = None) -> None:
        """Record a new user input and update pause duration."""
        now = time.time() if timestamp is None else timestamp
        if self._last_time is not None:
            self.pauses.append(now - self._last_time)
        self._last_time = now
        self.total_inputs += 1
        self._update_metrics()

    # --------------------------------------------------------------
    def log_correction(self) -> None:
        """Record a user correction event."""
        self.corrections += 1
        self._update_metrics()

    # --------------------------------------------------------------
    def _avg_pause(self) -> float:
        return float(sum(self.pauses) / len(self.pauses)) if self.pauses else 0.0

    # --------------------------------------------------------------
    def cognitive_load(self) -> float:
        """Return a 0–1 load index from pauses and corrections."""
        pause = min(self._avg_pause() / self.pause_threshold, 1.0)
        rate = self.corrections / self.total_inputs if self.total_inputs else 0.0
        return (pause + rate) / 2.0

    # --------------------------------------------------------------
    def _update_metrics(self) -> None:
        avg_pause = self._avg_pause()
        corr_rate = self.corrections / self.total_inputs if self.total_inputs else 0.0
        load = self.cognitive_load()
        if _HAS_PROM:
            assert isinstance(self.telemetry.metrics["avg_pause"], Gauge)
            self.telemetry.metrics["avg_pause"].set(avg_pause)
            self.telemetry.metrics["correction_rate"].set(corr_rate)
            self.telemetry.metrics["cognitive_load"].set(load)
        else:
            self.telemetry.metrics["avg_pause"] = avg_pause
            self.telemetry.metrics["correction_rate"] = corr_rate
            self.telemetry.metrics["cognitive_load"] = load
        for cb in list(self.callbacks):
            try:
                cb(load)
            except Exception:
                pass

    # --------------------------------------------------------------
    def get_metrics(self) -> Dict[str, float]:
        """Return current metrics as a dictionary."""
        return {
            "avg_pause": self._avg_pause(),
            "correction_rate": self.corrections / self.total_inputs if self.total_inputs else 0.0,
            "cognitive_load": self.cognitive_load(),
        }

    # --------------------------------------------------------------
    def stream_metrics(self) -> Iterable[Dict[str, float]]:
        """Yield metrics for each recorded interaction."""
        for i in range(max(len(self.pauses), 1)):
            yield self.get_metrics()


__all__ = ["CognitiveLoadMonitor"]
