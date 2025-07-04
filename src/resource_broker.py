from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .telemetry import TelemetryLogger


@dataclass
class Cluster:
    capacity: int
    used: int = 0
    accel_capacity: Dict[str, int] = field(default_factory=dict)
    accel_used: Dict[str, int] = field(default_factory=dict)


class ResourceBroker:
    """Coordinate jobs across multiple clusters."""

    def __init__(self) -> None:
        self.clusters: Dict[str, Cluster] = {}

    def register_cluster(self, name: str, capacity: int, accelerators: Dict[str, int] | None = None) -> None:
        self.clusters[name] = Cluster(capacity, 0, accelerators or {}, {k: 0 for k in (accelerators or {})})

    def allocate(self, job_name: str, accelerator: str = "cpu") -> str:
        """Return cluster assigned to ``job_name`` for ``accelerator``."""
        if not self.clusters:
            raise RuntimeError("no clusters registered")
        available = [
            (n, c)
            for n, c in self.clusters.items()
            if c.accel_capacity.get(accelerator, c.capacity if accelerator == "cpu" else 0)
            > c.accel_used.get(accelerator, 0)
        ]
        if not available:
            raise RuntimeError("no capacity for accelerator")
        chosen = min(available, key=lambda kv: kv[1].accel_used.get(accelerator, 0))[0]
        cl = self.clusters[chosen]
        cl.used += 1
        cl.accel_used[accelerator] = cl.accel_used.get(accelerator, 0) + 1
        return chosen

    def scale_decision(self, metrics: Dict[str, float]) -> str:
        util = max(metrics.get("cpu", 0.0), metrics.get("gpu", 0.0))
        if util > 80:
            return "scale_up"
        if util < 30:
            return "scale_down"
        return "stable"

    def get_load(self) -> Dict[str, float]:
        total = {"cpu": 0, "gpu": 0, "tpu": 0}
        used = {"cpu": 0, "gpu": 0, "tpu": 0}
        for cl in self.clusters.values():
            for acc in total.keys():
                cap = cl.accel_capacity.get(acc, cl.capacity if acc == "cpu" else 0)
                total[acc] += cap
                used[acc] += cl.accel_used.get(acc, 0)
        load = {}
        for acc in total:
            load[acc] = used[acc] / total[acc] if total[acc] else 0.0
        return load


__all__ = ["ResourceBroker"]
