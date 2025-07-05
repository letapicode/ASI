from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable, Any

from .telemetry import TelemetryLogger
from .reasoning_history import ReasoningHistoryLogger


class CollaborationPortal:
    """Expose tasks, telemetry metrics and reasoning logs over HTTP."""

    def __init__(
        self,
        tasks: Iterable[str] | None = None,
        telemetry: TelemetryLogger | None = None,
        reasoning: ReasoningHistoryLogger | None = None,
    ) -> None:
        self.tasks = list(tasks or [])
        self.telemetry = telemetry
        self.reasoning = reasoning
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

    # --------------------------------------------------------------
    def add_task(self, task: str) -> None:
        self.tasks.append(task)

    # --------------------------------------------------------------
    def complete_task(self, task: str) -> None:
        if task in self.tasks:
            self.tasks.remove(task)

    # --------------------------------------------------------------
    def get_tasks(self) -> list[str]:
        return list(self.tasks)

    # --------------------------------------------------------------
    def get_metrics(self) -> dict[str, Any]:
        return self.telemetry.get_stats() if self.telemetry else {}

    # --------------------------------------------------------------
    def get_logs(self) -> list[tuple[str, str]]:
        return self.reasoning.get_history() if self.reasoning else []

    # --------------------------------------------------------------
    def to_html(self) -> str:
        tasks = "".join(f"<li>{t}</li>" for t in self.tasks) or "<li>None</li>"
        logs = (
            "".join(f"<li>{ts}: {msg}</li>" for ts, msg in self.get_logs())
            or "<li>None</li>"
        )
        metrics = self.get_metrics()
        rows = "".join(
            f"<tr><td>{k}</td><td>{v:.3f}</td></tr>" for k, v in metrics.items()
        )
        return (
            "<html><body><h1>Collaboration Portal</h1>"
            "<h2>Active Tasks</h2><ul>" + tasks + "</ul>"
            "<h2>Telemetry</h2><table border='1'>"
            "<tr><th>Metric</th><th>Value</th></tr>" + rows + "</table>"
            "<h2>Reasoning Log</h2><ul>" + logs + "</ul>"
            "</body></html>"
        )

    # --------------------------------------------------------------
    def start(self, host: str = "localhost", port: int = 8070) -> None:
        if self.httpd is not None:
            return
        portal = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path == "/tasks":
                    data = json.dumps(portal.get_tasks()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/metrics":
                    data = json.dumps(portal.get_metrics()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/logs":
                    data = json.dumps(portal.get_logs()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    html = portal.to_html().encode()
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


__all__ = ["CollaborationPortal"]
