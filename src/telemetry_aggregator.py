"""HTTP service aggregating telemetry metrics from multiple nodes."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict


class TelemetryAggregator:
    """Collect metrics from ``TelemetryLogger`` nodes and expose Prometheus text."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, float]] = {}
        self.lock = threading.Lock()
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

    # --------------------------------------------------------------
    def ingest(self, data: Dict[str, Any]) -> None:
        node = str(data.get("node_id", "default"))
        with self.lock:
            metrics = self.nodes.setdefault(node, {})
            for k, v in data.items():
                if k == "node_id":
                    continue
                try:
                    metrics[k] = metrics.get(k, 0.0) + float(v)
                except Exception:
                    pass

    # --------------------------------------------------------------
    def aggregate(self) -> Dict[str, float]:
        totals = {
            "cpu": 0.0,
            "gpu": 0.0,
            "net": 0.0,
            "carbon_g": 0.0,
            "energy_kwh": 0.0,
            "prof_cpu_time": 0.0,
            "prof_gpu_mem": 0.0,
        }
        with self.lock:
            count = len(self.nodes)
            for stats in self.nodes.values():
                for k in totals:
                    if k in stats:
                        totals[k] += float(stats[k])
            if count:
                totals["cpu"] /= count
                totals["gpu"] /= count
        return totals

    # --------------------------------------------------------------
    def metrics_text(self) -> str:
        totals = self.aggregate()
        lines = [f"{k} {v}" for k, v in totals.items()]
        return "\n".join(lines)

    # --------------------------------------------------------------
    def start(self, host: str = "localhost", port: int = 8055) -> None:
        if self.httpd is not None:
            return
        aggregator = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: D401
                length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(length).decode() if length else "{}"
                try:
                    data = json.loads(payload) if payload else {}
                except Exception:
                    data = {}
                aggregator.ingest(data)
                self.send_response(200)
                self.end_headers()

            def do_GET(self) -> None:  # noqa: D401
                if self.path == "/metrics":
                    text = aggregator.metrics_text().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(text)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        self.httpd = HTTPServer((host, port), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    # --------------------------------------------------------------
    def stop(self) -> None:
        if self.httpd is None:
            return
        assert self.thread is not None
        self.httpd.shutdown()
        self.thread.join(timeout=1.0)
        self.httpd.server_close()
        self.httpd = None
        self.thread = None
        self.port = None


__all__ = ["TelemetryAggregator"]
