import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable, Dict, Any
import threading

from .risk_scoreboard import RiskScoreboard
from .memory_dashboard import MemoryDashboard


class RiskDashboard:
    """Serve combined risk and memory metrics."""

    def __init__(self, scoreboard: RiskScoreboard, servers: Iterable[Any], carbon_dashboard_url: str | None = None):
        self.scoreboard = scoreboard
        self.mem_dash = MemoryDashboard(servers)
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.carbon_url = carbon_dashboard_url

    def aggregate(self) -> Dict[str, float]:
        data = self.mem_dash.aggregate()
        data.update(self.scoreboard.get_metrics())
        if self.carbon_url:
            data["carbon_dashboard"] = self.carbon_url
        return data

    def start(self, host: str = "localhost", port: int = 8050) -> None:
        if self.httpd is not None:
            return
        dash = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/json":
                    out = json.dumps(dash.aggregate()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(out)
                else:
                    data = dash.aggregate()
                    link = (
                        f"<p><a href='{dash.carbon_url}'>Cluster Carbon Dashboard</a></p>"
                        if dash.carbon_url
                        else ""
                    )
                    html = (
                        "<html><body><pre>"
                        + json.dumps(data, indent=2)
                        + "</pre>" + link + "</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html.encode())

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        self.httpd = HTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd is not None:
            assert self.thread is not None
            self.httpd.shutdown()
            self.thread.join(timeout=1.0)
            self.httpd.server_close()
            self.httpd = None
            self.thread = None

__all__ = ["RiskDashboard"]
