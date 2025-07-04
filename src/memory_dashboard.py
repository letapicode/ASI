from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable, Dict, Any
import torch

from .retrieval_explainer import RetrievalExplainer

from .hierarchical_memory import MemoryServer


class MemoryDashboard:
    """Aggregate telemetry stats from multiple ``MemoryServer`` instances."""

    def __init__(self, servers: Iterable[MemoryServer]):
        self.servers = list(servers)
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None

    # ----------------------------------------------------------
    def aggregate(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for srv in self.servers:
            stats = srv.memory.get_stats()
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            if srv.telemetry is not None:
                tstats = srv.telemetry.get_stats()
                for k, v in tstats.items():
                    if isinstance(v, (int, float)):
                        totals[k] = totals.get(k, 0.0) + float(v)
        return totals

    # ----------------------------------------------------------
    def start(self, host: str = "localhost", port: int = 8050) -> None:
        if self.httpd is not None:
            return

        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path == "/trace" and dashboard.servers:
                    trace = dashboard.servers[0].memory.last_trace
                    if trace is None:
                        data = b"{}"
                    else:
                        vecs, meta, scores = (
                            None,
                            trace.get("provenance", []),
                            trace.get("scores", []),
                        )
                        data = json.dumps(
                            RetrievalExplainer.format(
                                torch.tensor([]), torch.tensor([]), scores, meta
                            )
                        ).encode()
                else:
                    data = json.dumps(dashboard.aggregate()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        self.httpd = HTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd is None:
            return
        assert self.thread is not None
        self.httpd.shutdown()
        self.thread.join(timeout=1.0)
        self.httpd.server_close()
        self.httpd = None
        self.thread = None


__all__ = ["MemoryDashboard"]
