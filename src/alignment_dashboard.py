from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Iterable, Dict, Any, Type

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


class AlignmentDashboard(BaseDashboard):
    """Aggregate alignment metrics and serve them via HTTP."""

    def __init__(self) -> None:
        super().__init__()
        self.total = 0
        self.passed = 0
        self.flagged: list[str] = []
        self.normative: list[str] = []
        self.bci_events = 0

    # --------------------------------------------------------------
    def record(
        self,
        passed: bool,
        flagged: Iterable[str] | None = None,
        normative: Iterable[str] | None = None,
        bci_events: int = 0,
    ) -> None:
        """Record a single alignment check result."""
        self.total += 1
        if passed:
            self.passed += 1
        if flagged:
            self.flagged.extend([str(f) for f in flagged])
        if normative:
            self.normative.extend([str(n) for n in normative])
        if bci_events:
            self.bci_events += int(bci_events)

    # --------------------------------------------------------------
    def aggregate(self) -> Dict[str, Any]:
        rate = self.passed / self.total if self.total else 0.0
        return {
            "total": float(self.total),
            "passed": float(self.passed),
            "pass_rate": float(rate),
            "flagged_examples": list(self.flagged[-10:]),
            "normative_violations": list(self.normative[-10:]),
            "bci_events": float(self.bci_events),
        }

    # --------------------------------------------------------------
    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                data = json.dumps(dashboard.aggregate()).encode()
                if self.path in ("/stats", "/json"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    html = (
                        "<html><body><pre>"
                        + json.dumps(dashboard.aggregate(), indent=2)
                        + "</pre></body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html.encode())

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        return Handler

    # --------------------------------------------------------------
    def start(self, host: str = "localhost", port: int = 8055) -> None:
        super().start(host, port)

    # --------------------------------------------------------------
    def stop(self) -> None:
        super().stop()


__all__ = ["AlignmentDashboard"]
