from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List

from .training_anomaly_detector import TrainingAnomalyDetector
from .risk_dashboard import RiskDashboard


@dataclass
class DistributedAnomalyMonitor:
    """Collect anomaly metrics across training nodes."""

    risk_dashboard: RiskDashboard | None = None
    detectors: Dict[str, TrainingAnomalyDetector] = field(default_factory=dict)
    anomaly_counts: Dict[str, int] = field(default_factory=dict)
    step_events: Dict[int, List[str]] = field(default_factory=dict)
    cross_run_events: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, node_id: str, step: int, loss: float) -> bool:
        """Record ``loss`` for ``node_id`` and return ``True`` if anomalous."""
        det = self.detectors.setdefault(node_id, TrainingAnomalyDetector())
        anomalous = det.record(loss)
        if anomalous:
            self.anomaly_counts[node_id] = self.anomaly_counts.get(node_id, 0) + 1
            nodes = self.step_events.setdefault(step, [])
            nodes.append(node_id)
            if len(nodes) >= 2 and all(e.get("step") != step for e in self.cross_run_events):
                event = {"step": step, "nodes": list(nodes)}
                self.cross_run_events.append(event)
                if self.risk_dashboard is not None:
                    self.risk_dashboard.scoreboard.metrics["cross_run_anomalies"] = float(
                        len(self.cross_run_events)
                    )
        return anomalous

    def aggregate(self) -> Dict[str, Any]:
        """Return combined anomaly statistics."""
        return {
            "total_anomalies": float(sum(self.anomaly_counts.values())),
            "per_node": {k: float(v) for k, v in self.anomaly_counts.items()},
            "cross_run_events": list(self.cross_run_events),
        }


__all__ = ["DistributedAnomalyMonitor"]
