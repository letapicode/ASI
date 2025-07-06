from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from .graph_of_thought import GraphOfThought
from .reasoning_history import ReasoningHistoryLogger
from .telemetry import TelemetryLogger


class IntrospectionDashboard:
    """Serve reasoning history merged with telemetry statistics."""

    def __init__(
        self,
        graph: GraphOfThought,
        history: ReasoningHistoryLogger,
        telemetry: TelemetryLogger,
    ) -> None:
        self.graph = graph
        self.history = history
        self.telemetry = telemetry
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

    # --------------------------------------------------------------
    def aggregate(self) -> Dict[str, Any]:
        """Return graph JSON, history log and telemetry metrics."""
        return {
            "graph": self.graph.to_json(),
            "history": self.history.get_history(),
            "telemetry": self.telemetry.get_stats(),
        }

    # --------------------------------------------------------------
    def to_html(self) -> str:
        data = self.aggregate()
        tele = data.get("telemetry", {})
        rows = "".join(
            f"<tr><td>{k}</td><td>{float(v):.2f}</td></tr>"
            for k, v in tele.items()
            if isinstance(v, (int, float))
        )
        hist_rows = "".join(
            f"<li>{ts}: {summary}</li>" for ts, summary in data.get("history", [])[-5:]
        )
        return (
            "<html><body><h1>Introspection Dashboard</h1>"
            "<h2>Telemetry</h2><table border='1'>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            f"{rows}</table>"
            "<h2>History</h2><ul>" + hist_rows + "</ul>"
            "</body></html>"
        )

    # --------------------------------------------------------------
    def start(self, host: str = "localhost", port: int = 8060) -> None:
        if self.httpd is not None:
            return
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path == "/graph":
                    data = json.dumps(dashboard.graph.to_json()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/history":
                    data = json.dumps(dashboard.history.get_history()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path in ("/stats", "/telemetry"):
                    data = json.dumps(dashboard.telemetry.get_stats()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/json":
                    data = json.dumps(dashboard.aggregate()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    html = dashboard.to_html().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html)

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


__all__ = ["IntrospectionDashboard"]

