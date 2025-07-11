from __future__ import annotations

"""Dashboard aggregating telemetry and reasoning logs from multiple agents."""

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Type

try:
    from .dashboard_import_helper import load_base_dashboard
except Exception:  # pragma: no cover - fallback when not packaged
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "dashboard_import_helper", Path(__file__).with_name("dashboard_import_helper.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    sys.modules.setdefault("dashboard_import_helper", module)
    load_base_dashboard = module.load_base_dashboard  # type: ignore

BaseDashboard = load_base_dashboard(__file__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .multi_agent_coordinator import MultiAgentCoordinator


class MultiAgentDashboard(BaseDashboard):
    """Aggregate telemetry and reasoning logs and serve them via HTTP."""

    def __init__(self, coordinator: "MultiAgentCoordinator | Any") -> None:
        super().__init__()
        self.coordinator = coordinator

    # --------------------------------------------------------------
    def aggregate(self) -> Dict[str, Any]:
        """Return combined metrics across all agents."""
        telemetry: Dict[str, float] = {}
        reasoning: Dict[str, list] = {}
        graphs: Dict[str, Any] = {}
        languages: Dict[str, list[str]] = {}
        for name, agent in self.coordinator.agents.items():
            tel = getattr(agent, "telemetry", None)
            hist = (
                getattr(agent, "history", None)
                or getattr(agent, "reasoning", None)
                or getattr(agent, "reasoning_log", None)
            )
            graph = (
                getattr(agent, "graph", None)
                or getattr(agent, "reasoning_graph", None)
                or getattr(agent, "got", None)
            )
            if tel is not None and hasattr(tel, "get_stats"):
                for k, v in tel.get_stats().items():
                    if isinstance(v, (int, float)):
                        telemetry[k] = telemetry.get(k, 0.0) + float(v)
            if hist is not None and hasattr(hist, "get_history"):
                reasoning[name] = list(hist.get_history())
                tr = getattr(hist, "translator", None)
                if tr is not None:
                    languages[name] = list(tr.languages)
            if graph is not None:
                graphs[name] = graph

        merged: Any | None = None
        inconsistencies: list | None = None
        if graphs:
            try:
                from .reasoning_merger import merge_graphs

                merged_graph, inconsistencies = merge_graphs(graphs)
                merged = merged_graph.to_json()
            except Exception:
                merged = None
                inconsistencies = None
        return {
            "assignments": list(self.coordinator.log),
            "telemetry": telemetry,
            "reasoning": reasoning,
            "merged_reasoning": merged,
            "inconsistencies": inconsistencies,
            "languages": languages,
        }

    # --------------------------------------------------------------
    def to_html(self) -> str:
        data = self.aggregate()
        tele = data.get("telemetry", {})
        carbon = tele.get("carbon_g", 0.0)
        langs = {
            name: ', '.join(l) for name, l in data.get("languages", {}).items()
        }
        rows = [
            f"<tr><td>{a}</td><td>{r}</td><td>{act}</td><td>{rew:.2f}</td></tr>"
            for a, r, act, rew in data.get("assignments", [])
        ]
        rrows = []
        for name, entries in data.get("reasoning", {}).items():
            for ts, summary in entries[-3:]:
                rrows.append(
                    f"<tr><td>{name}</td><td>{ts}</td><td>{summary}</td></tr>"
                )
        return (
            "<html><body><h1>Multi-Agent Dashboard</h1>"
            f"<p>Carbon emitted: {carbon:.2f} g</p>"
            "<h2>Languages</h2><ul>"
            + "".join(f"<li>{n}: {l}</li>" for n, l in langs.items())
            + "</ul>"
            "<h2>Task Log</h2><table border='1'>"
            "<tr><th>Agent</th><th>Repo</th><th>Action</th><th>Reward</th></tr>"
            f"{''.join(rows)}</table>"
            "<h2>Reasoning</h2><table border='1'>"
            "<tr><th>Agent</th><th>Time</th><th>Summary</th></tr>"
            f"{''.join(rrows)}</table></body></html>"
        )

    # --------------------------------------------------------------
    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
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

        return Handler

    def start(self, host: str = "localhost", port: int = 8070) -> None:
        super().start(host, port)

    # --------------------------------------------------------------
    def stop(self) -> None:
        super().stop()


__all__ = ["MultiAgentDashboard"]
