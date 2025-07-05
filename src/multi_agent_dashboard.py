from __future__ import annotations

"""Dashboard aggregating telemetry and reasoning logs from multiple agents."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from .multi_agent_coordinator import MultiAgentCoordinator


class MultiAgentDashboard:
    """Aggregate telemetry and reasoning logs and serve them via HTTP."""

    def __init__(self, coordinator: MultiAgentCoordinator) -> None:
        self.coordinator = coordinator
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

    # --------------------------------------------------------------
    def aggregate(self) -> Dict[str, Any]:
        """Return combined metrics across all agents."""
        telemetry: Dict[str, float] = {}
        reasoning: Dict[str, list] = {}
        for name, agent in self.coordinator.agents.items():
            tel = getattr(agent, "telemetry", None)
            hist = (
                getattr(agent, "history", None)
                or getattr(agent, "reasoning", None)
                or getattr(agent, "reasoning_log", None)
            )
            if tel is not None and hasattr(tel, "get_stats"):
                for k, v in tel.get_stats().items():
                    if isinstance(v, (int, float)):
                        telemetry[k] = telemetry.get(k, 0.0) + float(v)
            if hist is not None and hasattr(hist, "get_history"):
                reasoning[name] = list(hist.get_history())
        return {
            "assignments": list(self.coordinator.log),
            "telemetry": telemetry,
            "reasoning": reasoning,
        }

    # --------------------------------------------------------------
    def to_html(self) -> str:
        data = self.aggregate()
        tele = data.get("telemetry", {})
        carbon = tele.get("carbon_g", 0.0)
        rows = [
            f"<tr><td>{a}</td><td>{r}</td><td>{act}</td><td>{rew:.2f}</td></tr>"
            for a, r, act, rew in data.get("assignments", [])
        ]
        rrows = []
        for name, entries in data.get("reasoning", {}).items():
            for ts, summary in entries[-3:]:
                rrows.append(f"<tr><td>{name}</td><td>{ts}</td><td>{summary}</td></tr>")
        return (
            "<html><body><h1>Multi-Agent Dashboard</h1>"
            f"<p>Carbon emitted: {carbon:.2f} g</p>"
            "<h2>Task Log</h2><table border='1'>"
            "<tr><th>Agent</th><th>Repo</th><th>Action</th><th>Reward</th></tr>"
            f"{''.join(rows)}</table>"
            "<h2>Reasoning</h2><table border='1'>"
            "<tr><th>Agent</th><th>Time</th><th>Summary</th></tr>"
            f"{''.join(rrows)}</table></body></html>"
        )

    # --------------------------------------------------------------
    def start(self, host: str = "localhost", port: int = 8070) -> None:
        if self.httpd is not None:
            return
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path in ("/stats", "/json"):
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


__all__ = ["MultiAgentDashboard"]
