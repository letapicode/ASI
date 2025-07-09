import json
from http.server import BaseHTTPRequestHandler
from typing import Iterable, Dict, Any, Type
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

from .risk_scoreboard import RiskScoreboard
from .memory_dashboard import MemoryDashboard


class RiskDashboard(BaseDashboard):
    """Serve combined risk and memory metrics."""

    def __init__(self, scoreboard: RiskScoreboard, servers: Iterable[Any], carbon_dashboard_url: str | None = None):
        super().__init__()
        self.scoreboard = scoreboard
        self.mem_dash = MemoryDashboard(servers)
        self.carbon_url = carbon_dashboard_url

    def aggregate(self) -> Dict[str, float]:
        data = self.mem_dash.aggregate()
        data.update(self.scoreboard.get_metrics())
        if self.carbon_url:
            data["carbon_dashboard"] = self.carbon_url
        return data

    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
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

        return Handler

    def start(self, host: str = "localhost", port: int = 8050) -> None:
        super().start(host, port)

    def stop(self) -> None:
        super().stop()

__all__ = ["RiskDashboard"]
