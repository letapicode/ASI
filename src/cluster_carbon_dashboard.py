from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Dict, Any, Type
import importlib.util
import sys
from pathlib import Path

try:
    from .dashboard_base import BaseDashboard
except Exception:  # pragma: no cover - fallback when not packaged
    spec = importlib.util.spec_from_file_location(
        "dashboard_base", Path(__file__).with_name("dashboard_base.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    sys.modules.setdefault("dashboard_base", module)
    BaseDashboard = module.BaseDashboard  # type: ignore


class ClusterCarbonDashboard(BaseDashboard):
    """Aggregate carbon metrics from multiple nodes."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.schedules: list[tuple[str, float]] = []

    # --------------------------------------------------------------
    def record(
        self,
        node_id: str,
        energy: float,
        carbon: float,
        intensity: float | None = None,
        cost: float | None = None,
    ) -> None:
        rec = self.metrics.setdefault(
            node_id,
            {
                "energy_kwh": 0.0,
                "carbon_g": 0.0,
                "carbon_intensity": 0.0,
                "energy_cost": 0.0,
            },
        )
        rec["energy_kwh"] += float(energy)
        rec["carbon_g"] += float(carbon)
        if intensity is not None:
            rec["carbon_intensity"] = float(intensity)
        if cost is not None:
            rec["energy_cost"] += float(cost)
        self.metrics[node_id] = rec

    def record_schedule(self, cluster: str, carbon_saved: float) -> None:
        self.schedules.append((cluster, float(carbon_saved)))

    def aggregate(self) -> Dict[str, Any]:
        total = {"energy_kwh": 0.0, "carbon_g": 0.0, "energy_cost": 0.0}
        for m in self.metrics.values():
            total["energy_kwh"] += m["energy_kwh"]
            total["carbon_g"] += m["carbon_g"]
            total["energy_cost"] += m.get("energy_cost", 0.0)
        if total["energy_kwh"]:
            total["carbon_intensity"] = total["carbon_g"] / total["energy_kwh"]
        saved = sum(s for _, s in self.schedules)
        return {"total": total, "nodes": self.metrics, "schedules": self.schedules, "carbon_saved": saved}

    # --------------------------------------------------------------
    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: D401
                if self.path == "/update":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length) if length else b"{}"
                    data = json.loads(body.decode() or "{}")
                    node = str(data.get("node_id"))
                    energy = float(data.get("energy_kwh", 0.0))
                    carbon = float(data.get("carbon_g", 0.0))
                    intensity = float(data.get("carbon_intensity", 0.0))
                    cost = float(data.get("energy_cost", 0.0))
                    dashboard.record(node, energy, carbon, intensity, cost)
                    self.send_response(200)
                    self.end_headers()
                elif self.path == "/schedule":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length) if length else b"{}"
                    data = json.loads(body.decode() or "{}")
                    cl = str(data.get("cluster"))
                    saved = float(data.get("carbon_saved", 0.0))
                    dashboard.record_schedule(cl, saved)
                    self.send_response(200)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_GET(self) -> None:  # noqa: D401
                if self.path in ("/stats", "/json"):
                    data = json.dumps(dashboard.aggregate()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    agg = dashboard.aggregate()
                    rows = []
                    for node, m in dashboard.metrics.items():
                        rows.append(
                            f"<tr><td>{node}</td><td>{m['energy_kwh']:.3f}</td><td>{m['carbon_g']:.3f}</td><td>{m.get('carbon_intensity',0.0):.3f}</td><td>{m.get('energy_cost',0.0):.3f}</td></tr>"
                        )
                    rows_str = "\n".join(rows)
                    total = agg["total"]
                    sched_rows = "".join(
                        f"<tr><td>{c}</td><td>{s:.3f}</td></tr>" for c, s in dashboard.schedules[-10:]
                    )
                    carbon_saved = sum(s for _, s in dashboard.schedules)
                    html = (
                        "<html><body><h1>Cluster Carbon Dashboard</h1>"
                        "<table border='1'><tr><th>Node</th><th>Energy (kWh)</th><th>Carbon (g)</th><th>Intensity</th><th>Cost</th></tr>"
                        f"{rows_str}</table>"
                        f"<p>Total energy: {total['energy_kwh']:.3f} kWh, Carbon: {total['carbon_g']:.3f} g, Cost: ${total['energy_cost']:.2f}</p>"
                        "<h2>Schedules</h2><table border='1'><tr><th>Cluster</th><th>Carbon Saved</th></tr>"
                        f"{sched_rows}</table>"
                        f"<p>Total carbon saved: {carbon_saved:.3f} g</p>"
                        "</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html.encode())

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        return Handler

    def start(self, host: str = "localhost", port: int = 8090) -> None:
        super().start(host, port)

    def stop(self) -> None:
        super().stop()


__all__ = ["ClusterCarbonDashboard"]
