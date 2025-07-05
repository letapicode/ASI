"""Demo of DistributedAnomalyMonitor."""

from __future__ import annotations

import random

from asi.training_anomaly_detector import TrainingAnomalyDetector
from asi.risk_scoreboard import RiskScoreboard
from asi.risk_dashboard import RiskDashboard
from asi.distributed_anomaly_monitor import DistributedAnomalyMonitor


def main() -> None:
    board = RiskScoreboard()
    dash = RiskDashboard(board, [])
    detectors = {
        "node1": TrainingAnomalyDetector(window=2, threshold=1.5),
        "node2": TrainingAnomalyDetector(window=2, threshold=1.5),
    }
    monitor = DistributedAnomalyMonitor(dash, detectors)

    for step in range(5):
        for node in ("node1", "node2"):
            loss = random.uniform(0.5, 3.0)
            monitor.record(node, step, loss)

    print("Aggregated", monitor.aggregate())


if __name__ == "__main__":  # pragma: no cover - demo
    main()
