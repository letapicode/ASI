from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable, Dict, Any
import numpy as np
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
        gpu_vals = []
        score_vals = []
        for srv in self.servers:
            stats = srv.memory.get_stats()
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            trace = getattr(srv.memory, "last_trace", None)
            if trace is not None and trace.get("scores"):
                avg_score = float(np.mean(trace["scores"]))
                totals["model_score"] = totals.get("model_score", 0.0) + avg_score
                score_vals.append(avg_score)
            if srv.telemetry is not None:
                tstats = srv.telemetry.get_stats()
                for k, v in tstats.items():
                    if isinstance(v, (int, float)):
                        totals[k] = totals.get(k, 0.0) + float(v)
                gpu = tstats.get("gpu")
                if gpu is not None and trace is not None and trace.get("scores"):
                    gpu_vals.append(float(gpu))
        hits = totals.get("hits", 0.0)
        misses = totals.get("misses", 0.0)
        total = hits + misses
        if total:
            totals["hit_rate"] = hits / total
        if len(gpu_vals) >= 2 and len(score_vals) >= 2:
            totals["gpu_score_corr"] = float(np.corrcoef(gpu_vals, score_vals)[0, 1])
        else:
            totals["gpu_score_corr"] = 0.0
        return totals

    # ----------------------------------------------------------
    def to_html(self) -> str:
        rows = []
        for idx, srv in enumerate(self.servers):
            tstats = srv.telemetry.get_stats() if srv.telemetry else {}
            gpu = tstats.get("gpu", 0.0)
            mstats = srv.memory.get_stats()
            hits = int(mstats.get("hits", 0))
            misses = int(mstats.get("misses", 0))
            score = 0.0
            trace = getattr(srv.memory, "last_trace", None)
            if trace is not None and trace.get("scores"):
                score = float(np.mean(trace["scores"]))
            rows.append(
                f"<tr><td>{idx}</td><td>{gpu:.2f}</td><td>{hits}</td><td>{misses}</td><td>{score:.3f}</td></tr>"
            )
        corr = self.aggregate().get("gpu_score_corr", 0.0)
        table = "\n".join(rows)
        return (
            "<html><body><h1>Memory Dashboard</h1>"
            "<table border='1'>"
            "<tr><th>Server</th><th>GPU Util (%)</th><th>Hits</th><th>Misses</th><th>Avg Score</th></tr>"
            f"{table}</table><p>GPU/Score correlation: {corr:.3f}</p></body></html>"
        )

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
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(data)
                        return
                elif self.path in ("/stats", "/json"):
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
