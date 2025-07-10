from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Iterable, Dict, Any, Type
import base64
import numpy as np
import torch

from .retrieval_explainer import RetrievalExplainer
from .retrieval_visualizer import RetrievalVisualizer
from .retrieval_trust_scorer import RetrievalTrustScorer
from .memory_timeline_viewer import MemoryTimelineViewer
from .kg_visualizer import KGVisualizer

from .hierarchical_memory import MemoryServer

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


class MemoryDashboard(BaseDashboard):
    """Aggregate telemetry stats from multiple ``MemoryServer`` instances."""

    def __init__(
        self,
        servers: Iterable[MemoryServer],
        visualizer: RetrievalVisualizer | None = None,
        trust_scorer: "RetrievalTrustScorer | None" = None,
    ) -> None:
        super().__init__()
        self.servers = list(servers)
        self.visualizer = visualizer
        self.trust_scorer = trust_scorer
        self._fairness_img: str | None = None

    # ----------------------------------------------------------
    def _load_fairness(self) -> None:
        if self._fairness_img is not None:
            return
        root = Path(getattr(self.servers[0].memory, "dataset_root", "")) if self.servers else None
        if root and root.name:
            fname = f"{root.name}_fairness.png"
            path = Path("docs/datasets") / fname
            if path.exists():
                b64 = base64.b64encode(path.read_bytes()).decode()
                self._fairness_img = f"data:image/png;base64,{b64}"
                return
        self._fairness_img = ""

    # ----------------------------------------------------------
    def aggregate(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        gpu_vals = []
        score_vals = []
        trust_vals = []
        for srv in self.servers:
            stats = srv.memory.get_stats()
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            trace = getattr(srv.memory, "last_trace", None)
            if trace is not None and trace.get("scores"):
                avg_score = float(np.mean(trace["scores"]))
                totals["model_score"] = totals.get("model_score", 0.0) + avg_score
                score_vals.append(avg_score)
                if self.trust_scorer is not None:
                    ts = self.trust_scorer.score_results(trace.get("provenance", []))
                    if ts:
                        tavg = float(np.mean(ts))
                        totals["trust_score"] = totals.get("trust_score", 0.0) + tavg
                        trust_vals.append(tavg)
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
        if trust_vals:
            totals["trust_score"] = float(np.mean(trust_vals))
        return totals

    # ----------------------------------------------------------
    def _entries(self, start: int = 0, end: int | None = None) -> list[dict]:
        if not self.servers:
            return []
        metas = getattr(self.servers[0].memory.store, "_meta", [])
        if end is None or end > len(metas):
            end = len(metas)
        start = max(0, start)
        return [{"index": i, "meta": metas[i]} for i in range(start, end)]

    # ----------------------------------------------------------
    def events(self) -> list[dict]:
        all_events: list[dict] = []
        for srv in self.servers:
            if srv.telemetry is not None:
                all_events.extend(srv.telemetry.get_events())
        return all_events

    # ----------------------------------------------------------
    def pattern_image(self) -> str:
        if self.visualizer is None:
            return ""
        return self.visualizer.pattern_image()

    # ----------------------------------------------------------
    def to_html(self) -> str:
        self._load_fairness()
        rows = []
        for idx, srv in enumerate(self.servers):
            tstats = srv.telemetry.get_stats() if srv.telemetry else {}
            gpu = tstats.get("gpu", 0.0)
            mstats = srv.memory.get_stats()
            hits = int(mstats.get("hits", 0))
            misses = int(mstats.get("misses", 0))
            score = 0.0
            trust = 0.0
            trace = getattr(srv.memory, "last_trace", None)
            if trace is not None and trace.get("scores"):
                score = float(np.mean(trace["scores"]))
                if self.trust_scorer is not None:
                    ts = self.trust_scorer.score_results(trace.get("provenance", []))
                    if ts:
                        trust = float(np.mean(ts))
                        if srv.telemetry is not None:
                            srv.telemetry.record_trust(trust)
            rows.append(
                f"<tr><td>{idx}</td><td>{gpu:.2f}</td><td>{hits}</td><td>{misses}</td><td>{score:.3f}</td><td>{trust:.3f}</td></tr>"
            )
        corr = self.aggregate().get("gpu_score_corr", 0.0)
        table = "\n".join(rows)
        events = "".join(
            f"<li>{e['metric']} spike at {e['index']}</li>" for e in self.events()[-10:]
        )
        summary = ""
        if self.servers:
            trace = getattr(self.servers[0].memory, "last_trace", None)
            if trace is not None:
                summary = trace.get("summary", "")
        img = f"<img src='{self._fairness_img}'>" if self._fairness_img else ""
        return (
            f"<html><body><h1>Memory Dashboard</h1>{img}"
            "<p><a href='http://localhost:8070/graph'>Graph UI</a> | "
            "<a href='/kg'>KG Visualizer</a></p>"
            "<table border='1'>"
            "<tr><th>Server</th><th>GPU Util (%)</th><th>Hits</th><th>Misses</th><th>Avg Score</th><th>Avg Trust</th></tr>"
            f"{table}</table><p>GPU/Score correlation: {corr:.3f}</p>"
            + (f"<p>Last retrieval: {summary}</p>" if summary else "")
            + f"<h2>Events</h2><ul>{events}</ul></body></html>"
        )

    # ----------------------------------------------------------
    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path == "/trace" and dashboard.servers:
                    trace = dashboard.servers[0].memory.last_trace
                    if trace is None:
                        data = b"{}"
                    else:
                        q = torch.tensor(trace.get("query", []))
                        r = torch.tensor(trace.get("results", []))
                        meta = trace.get("provenance", [])
                        scores = trace.get("scores", [])
                        items = RetrievalExplainer.format(q, r, scores, meta)
                        trust = []
                        if dashboard.trust_scorer is not None:
                            trust = dashboard.trust_scorer.score_results(meta)
                            for it, t in zip(items, trust):
                                it["trust"] = t
                        summary = trace.get("summary")
                        if summary is None:
                            is_multi = any(
                                isinstance(m, dict)
                                and any(k in m for k in ("text", "image", "audio"))
                                for m in meta
                            ) if meta else False
                            if is_multi and hasattr(RetrievalExplainer, "summarize_multimodal"):
                                summary = RetrievalExplainer.summarize_multimodal(q, r, scores, meta)
                            else:
                                summary = RetrievalExplainer.summarize(q, r, scores, meta)
                        data = json.dumps({"items": items, "summary": summary, "trust": trust}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                    return
                elif self.path.startswith("/entries"):
                    q = parse_qs(urlparse(self.path).query)
                    start = int(q.get("start", [0])[0])
                    end = q.get("end", [None])[0]
                    end_i = int(end) if end is not None else None
                    data = json.dumps(dashboard._entries(start, end_i)).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/events":
                    data = json.dumps(dashboard.events()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/patterns":
                    img = dashboard.pattern_image()
                    html = f"<html><body><img src='{img}'></body></html>".encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html)
                elif self.path == "/timeline":
                    viewer = None
                    if dashboard.servers and dashboard.servers[0].telemetry is not None:
                        viewer = MemoryTimelineViewer(dashboard.servers[0].telemetry)
                    data = viewer.to_json().encode() if viewer else b"{}"
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/kg" or self.path == "/kg/":
                    kg = None
                    if dashboard.servers and getattr(dashboard.servers[0].memory, "kg", None) is not None:
                        kg = dashboard.servers[0].memory.kg
                    html = KGVisualizer(kg).to_html().encode() if kg else b"<html><body>No knowledge graph</body></html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html)
                elif self.path == "/kg/graph":
                    kg = None
                    if dashboard.servers and getattr(dashboard.servers[0].memory, "kg", None) is not None:
                        kg = dashboard.servers[0].memory.kg
                    data = json.dumps(KGVisualizer(kg).graph_json()).encode() if kg else b"{}"
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
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

            def do_POST(self) -> None:  # noqa: D401
                if self.path == "/add" and dashboard.servers:
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length) if length else b"{}"
                    data = json.loads(body.decode() or "{}")
                    vec = torch.tensor(data.get("vector", []), dtype=torch.float32)
                    if vec.ndim == 1:
                        vec = vec.unsqueeze(0)
                    meta = data.get("metadata")
                    dashboard.servers[0].memory.add(vec, metadata=[meta] if meta is not None else None)
                    out = json.dumps({"ok": True}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(out)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_DELETE(self) -> None:  # noqa: D401
                if self.path == "/delete" and dashboard.servers:
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length) if length else b"{}"
                    data = json.loads(body.decode() or "{}")
                    idx = data.get("index")
                    tag = data.get("tag")
                    confirm = data.get("confirm")
                    if idx is None and tag is None:
                        self.send_response(400)
                        self.end_headers()
                        return
                    n_del = 1
                    if isinstance(idx, list):
                        n_del = len(idx)
                    if idx is None and tag is not None:
                        metas = getattr(dashboard.servers[0].memory.store, "_meta", [])
                        n_del = sum(1 for m in metas if m == tag)
                    if n_del > 10 and confirm != "yes":
                        self.send_response(403)
                        self.end_headers()
                        return
                    dashboard.servers[0].memory.delete(index=idx, tag=tag)
                    out = json.dumps({"ok": True}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(out)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        return Handler

    def start(self, host: str = "localhost", port: int = 8050) -> None:
        super().start(host, port)

    def stop(self) -> None:
        super().stop()


__all__ = ["MemoryDashboard"]
