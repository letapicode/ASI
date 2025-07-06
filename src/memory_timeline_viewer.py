from __future__ import annotations

import json
from typing import Dict, Any, List

from .telemetry import TelemetryLogger
from .memory_event_detector import MemoryEventDetector


class MemoryTimelineViewer:
    """Build a JSON timeline from telemetry history and events."""

    def __init__(self, telemetry: TelemetryLogger) -> None:
        self.telemetry = telemetry
        # keep reference to the event detector for convenience
        self.detector: MemoryEventDetector = telemetry.event_detector

    # --------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        """Return timeline and events as a dictionary."""
        timeline: List[Dict[str, float]] = []
        hits = 0.0
        misses = 0.0
        for idx, snap in enumerate(self.telemetry.history):
            hits += float(snap.get("hits", 0.0))
            misses += float(snap.get("misses", 0.0))
            total = hits + misses
            hit_rate = hits / total if total else 0.0
            latency = float(snap.get("latency", 0.0))
            intensity = float(snap.get("carbon_intensity", 0.0))
            ts = idx * float(getattr(self.telemetry, "interval", 1.0))
            timeline.append(
                {
                    "timestamp": ts,
                    "hit_rate": hit_rate,
                    "latency": latency,
                    "carbon_intensity": intensity,
                }
            )
        return {"timeline": timeline, "events": list(self.telemetry.get_events())}

    # --------------------------------------------------------------
    def to_json(self) -> str:
        """Return ``build()`` result encoded as JSON."""
        return json.dumps(self.build())


__all__ = ["MemoryTimelineViewer"]
