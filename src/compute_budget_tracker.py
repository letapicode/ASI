from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict

try:  # pragma: no cover - allow running as standalone module
    from .telemetry import TelemetryLogger
except Exception:  # pragma: no cover - fallback for direct import
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _tel_path = _Path(__file__).resolve().parent / "telemetry.py"
    import sys as _sys

    _spec = _ilu.spec_from_file_location("telemetry", _tel_path)
    _mod = _ilu.module_from_spec(_spec)
    _mod.__package__ = "asi"
    _sys.modules["telemetry"] = _mod
    assert _spec and _spec.loader
    _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
    TelemetryLogger = _mod.TelemetryLogger


@dataclass
class ComputeBudgetTracker:
    """Track compute usage against a GPU-hour budget."""

    budget_hours: float
    telemetry: TelemetryLogger | None = None
    energy_per_gpu_hour: float = 0.3
    co2_per_kwh: float = 400.0
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
                run_id,
                {
                    "gpu_hours": 0.0,
                    "mem_peak": 0.0,
                    "energy": 0.0,
                    "carbon": 0.0,
                },
            )
            rec["gpu_hours"] += stats.get("gpu", 0.0) / 100.0 * interval / 3600
            rec["mem_peak"] = max(rec["mem_peak"], stats.get("mem", 0.0))
            if self.telemetry.carbon_tracker is not None:
                cf = self.telemetry.carbon_tracker.get_stats()
                energy_base = rec.get("_energy_base")
                if energy_base is None:
                    rec["_energy_base"] = cf.get("energy_kwh", 0.0)
                    rec["_carbon_base"] = cf.get("carbon_g", 0.0)
                rec["energy"] = cf.get("energy_kwh", 0.0) - rec.get("_energy_base", 0.0)
                rec["carbon"] = cf.get("carbon_g", 0.0) - rec.get("_carbon_base", 0.0)
            else:
                rec["energy"] = rec["gpu_hours"] * self.energy_per_gpu_hour
                rec["carbon"] = rec["energy"] * self.co2_per_kwh
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
        rec = dict(self.records.get(run_id, {}))
        rec.pop("_energy_base", None)
        rec.pop("_carbon_base", None)
        return rec

    def consume(self, run_id: str, gpu_hours: float, mem: float) -> None:
        """Manually log ``gpu_hours`` and peak ``mem`` for ``run_id``."""
        rec = self.records.setdefault(
            run_id,
            {
                "gpu_hours": 0.0,
                "mem_peak": 0.0,
                "energy": 0.0,
                "carbon": 0.0,
            },
        )
        rec["gpu_hours"] += gpu_hours
        rec["mem_peak"] = max(rec["mem_peak"], mem)
        rec["energy"] = rec["gpu_hours"] * self.energy_per_gpu_hour
        rec["carbon"] = rec["energy"] * self.co2_per_kwh
        self.records[run_id] = rec


__all__ = ["ComputeBudgetTracker"]

