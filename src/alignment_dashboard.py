from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable, Dict, Any


class AlignmentDashboard:
    """Aggregate alignment metrics and serve them via HTTP."""

    def __init__(self) -> None:
        self.total = 0
        self.passed = 0
        self.flagged: list[str] = []
        self.normative: list[str] = []
        self.bci_events = 0
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

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
    def start(self, host: str = "localhost", port: int = 8055) -> None:
        if self.httpd is not None:
            return
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


__all__ = ["AlignmentDashboard"]
