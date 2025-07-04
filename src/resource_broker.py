from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .telemetry import TelemetryLogger


@dataclass
class Cluster:
    capacity: int
    used: int = 0


class ResourceBroker:
    """Coordinate jobs across multiple clusters."""

    def __init__(self) -> None:
        self.clusters: Dict[str, Cluster] = {}

    def register_cluster(self, name: str, capacity: int) -> None:
        self.clusters[name] = Cluster(capacity)

    def allocate(self, job_name: str) -> str:
        """Return cluster assigned to ``job_name``."""
        if not self.clusters:
            raise RuntimeError("no clusters registered")
        chosen = min(self.clusters.items(), key=lambda kv: kv[1].used)[0]
        self.clusters[chosen].used += 1
        return chosen

    def scale_decision(self, metrics: Dict[str, float]) -> str:
        util = max(metrics.get("cpu", 0.0), metrics.get("gpu", 0.0))
        if util > 80:
            return "scale_up"
        if util < 30:
            return "scale_down"
        return "stable"


__all__ = ["ResourceBroker"]
