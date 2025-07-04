from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict

from .telemetry import TelemetryLogger


@dataclass
class ComputeBudgetTracker:
    """Track compute usage against a GPU-hour budget."""

    budget_hours: float
    telemetry: TelemetryLogger | None = None
    energy_per_gpu_hour: float = 0.3
    records: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.telemetry is None:
            self.telemetry = TelemetryLogger(interval=1.0)
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None

    # --------------------------------------------------
    def _collect(self, run_id: str) -> None:
        interval = self.telemetry.interval
        while not self._stop.is_set():
            stats = self.telemetry.get_stats()
            rec = self.records.setdefault(
                run_id, {"gpu_hours": 0.0, "mem_peak": 0.0, "energy": 0.0}
            )
            rec["gpu_hours"] += stats.get("gpu", 0.0) / 100.0 * interval / 3600
            rec["mem_peak"] = max(rec["mem_peak"], stats.get("mem", 0.0))
            rec["energy"] = rec["gpu_hours"] * self.energy_per_gpu_hour
            self.records[run_id] = rec
            if rec["gpu_hours"] >= self.budget_hours:
                self._stop.set()
            time.sleep(interval)

    # --------------------------------------------------
    def start(self, run_id: str) -> None:
        self.telemetry.start()
        self._stop.clear()
        self.thread = threading.Thread(target=self._collect, args=(run_id,), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        self.telemetry.stop()

    # --------------------------------------------------
    def remaining(self, run_id: str) -> float:
        rec = self.records.get(run_id, {"gpu_hours": 0.0})
        return max(self.budget_hours - rec["gpu_hours"], 0.0)

    def get_usage(self, run_id: str) -> Dict[str, float]:
        return dict(self.records.get(run_id, {}))

