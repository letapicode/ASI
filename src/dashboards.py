from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Iterable, Dict, Any, Type
import base64
try:
    import numpy as np
except Exception:  # pragma: no cover - allow running without numpy
    np = None  # type: ignore
try:
    import torch
except Exception:  # pragma: no cover - allow running without torch
    torch = None  # type: ignore

from .retrieval_analysis import RetrievalExplainer, RetrievalVisualizer
try:
    from .retrieval_trust_scorer import RetrievalTrustScorer
except Exception:  # pragma: no cover - optional dependency
    RetrievalTrustScorer = None  # type: ignore
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

from .graph_of_thought import GraphOfThought
from .reasoning_history import ReasoningHistoryLogger
from .telemetry import TelemetryLogger


class IntrospectionDashboard(BaseDashboard):
    """Serve reasoning history merged with telemetry statistics."""

    def __init__(
        self,
        graph: GraphOfThought,
        history: ReasoningHistoryLogger,
        telemetry: TelemetryLogger,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.history = history
        self.telemetry = telemetry

    # --------------------------------------------------------------
    def aggregate(self) -> Dict[str, Any]:
        """Return graph JSON, history log and telemetry metrics."""
        return {
            "graph": self.graph.to_json(),
            "history": self.history.get_history(),
            "telemetry": self.telemetry.get_stats(),
        }

    # --------------------------------------------------------------
    def to_html(self) -> str:
        data = self.aggregate()
        tele = data.get("telemetry", {})
        rows = "".join(
            f"<tr><td>{k}</td><td>{float(v):.2f}</td></tr>"
            for k, v in tele.items()
            if isinstance(v, (int, float))
        )
        hist_rows = "".join(
            f"<li>{ts}: {summary}</li>" for ts, summary in data.get("history", [])[-5:]
        )
        return (
            "<html><body><h1>Introspection Dashboard</h1>"
            "<h2>Telemetry</h2><table border='1'>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            f"{rows}</table>"
            "<h2>History</h2><ul>" + hist_rows + "</ul>"
            "</body></html>"
        )

    # --------------------------------------------------------------
    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path == "/graph":
                    data = json.dumps(dashboard.graph.to_json()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/history":
                    data = json.dumps(dashboard.history.get_history()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path in ("/stats", "/telemetry"):
                    data = json.dumps(dashboard.telemetry.get_stats()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/json":
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

    def start(self, host: str = "localhost", port: int = 8060) -> None:
        super().start(host, port)

    # --------------------------------------------------------------
    def stop(self) -> None:
        super().stop()







import base64
import json
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Iterable, Any, Dict, Type

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

from .graph_of_thought import GraphOfThought

try:  # pragma: no cover - optional heavy dep
    import torch
except Exception:  # pragma: no cover - fallback when torch is missing
    torch = None  # type: ignore

from .transformer_circuits import AttentionVisualizer

BaseDashboard = load_base_dashboard(__file__)

_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset='utf-8'><title>Interpretability Dashboard</title></head>
<body>
<h1>Interpretability Dashboard</h1>
<div id='stats'></div>
<div id='heatmaps'></div>
<script>
async function load() {
  const stats = await fetch('/stats').then(r => r.json());
  document.getElementById('stats').innerText = JSON.stringify(stats);
  const hmaps = await fetch('/heatmaps').then(r => r.json());
  for (const src of hmaps.images) {
    const img = document.createElement('img');
    img.src = src;
    img.style.margin = '2px';
    document.getElementById('heatmaps').appendChild(img);
  }
}
load();
</script>
</body>
</html>
"""


class InterpretabilityDashboard(BaseDashboard):
    """Serve attention heatmaps and memory statistics."""

    def __init__(
        self,
        model: "torch.nn.Module | Any",
        servers: Iterable[Any],
        sample: "torch.Tensor | Any",
        graph: GraphOfThought | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.mem_dash = MemoryDashboard(servers)
        self.graph = graph
        self.tmpdir = Path(tempfile.mkdtemp(prefix="attn_vis_"))
        if torch is not None:
            self.vis = AttentionVisualizer(model, out_dir=str(self.tmpdir))
            self.vis.run(sample)
        else:  # pragma: no cover - torch optional
            self.vis = None

    # --------------------------------------------------------------
    def _heatmaps(self) -> Dict[str, Any]:
        if self.vis is None:
            return {"images": []}
        imgs = []
        for p in self.vis.out_dir.glob("*.png"):
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            imgs.append(f"data:image/png;base64,{b64}")
        return {"images": imgs}

    # --------------------------------------------------------------
    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path == "/stats":
                    stats = dashboard.mem_dash.aggregate()
                    if dashboard.graph is not None:
                        contribs = {
                            str(nid): node.metadata.get("head_importance")
                            for nid, node in dashboard.graph.nodes.items()
                            if node.metadata and "head_importance" in node.metadata
                        }
                        if contribs:
                            stats["head_contributions"] = contribs
                    data = json.dumps(stats).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/heatmaps":
                    data = json.dumps(dashboard._heatmaps()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(_HTML.encode())

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        return Handler

    def start(self, host: str = "localhost", port: int = 8060) -> None:
        super().start(host, port)

    # --------------------------------------------------------------
    def stop(self) -> None:
        super().stop()





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

from .risk_scoreboard import RiskScoreboard


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





import json
from http.server import BaseHTTPRequestHandler
from typing import Dict, Any, Type

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


class ClusterCarbonDashboard(BaseDashboard):
    """Aggregate carbon metrics from multiple nodes."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.schedules: list[tuple[str, float]] = []

    # --------------------------------------------------------------
    def record(
        self,
        node_id: str,
        energy: float,
        carbon: float,
        intensity: float | None = None,
        cost: float | None = None,
    ) -> None:
        rec = self.metrics.setdefault(
            node_id,
            {
                "energy_kwh": 0.0,
                "carbon_g": 0.0,
                "carbon_intensity": 0.0,
                "energy_cost": 0.0,
            },
        )
        rec["energy_kwh"] += float(energy)
        rec["carbon_g"] += float(carbon)
        if intensity is not None:
            rec["carbon_intensity"] = float(intensity)
        if cost is not None:
            rec["energy_cost"] += float(cost)
        self.metrics[node_id] = rec

    def record_schedule(self, cluster: str, carbon_saved: float) -> None:
        self.schedules.append((cluster, float(carbon_saved)))

    def aggregate(self) -> Dict[str, Any]:
        total = {"energy_kwh": 0.0, "carbon_g": 0.0, "energy_cost": 0.0}
        for m in self.metrics.values():
            total["energy_kwh"] += m["energy_kwh"]
            total["carbon_g"] += m["carbon_g"]
            total["energy_cost"] += m.get("energy_cost", 0.0)
        if total["energy_kwh"]:
            total["carbon_intensity"] = total["carbon_g"] / total["energy_kwh"]
        saved = sum(s for _, s in self.schedules)
        return {"total": total, "nodes": self.metrics, "schedules": self.schedules, "carbon_saved": saved}

    # --------------------------------------------------------------
    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: D401
                if self.path == "/update":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length) if length else b"{}"
                    data = json.loads(body.decode() or "{}")
                    node = str(data.get("node_id"))
                    energy = float(data.get("energy_kwh", 0.0))
                    carbon = float(data.get("carbon_g", 0.0))
                    intensity = float(data.get("carbon_intensity", 0.0))
                    cost = float(data.get("energy_cost", 0.0))
                    dashboard.record(node, energy, carbon, intensity, cost)
                    self.send_response(200)
                    self.end_headers()
                elif self.path == "/schedule":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length) if length else b"{}"
                    data = json.loads(body.decode() or "{}")
                    cl = str(data.get("cluster"))
                    saved = float(data.get("carbon_saved", 0.0))
                    dashboard.record_schedule(cl, saved)
                    self.send_response(200)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_GET(self) -> None:  # noqa: D401
                if self.path in ("/stats", "/json"):
                    data = json.dumps(dashboard.aggregate()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    agg = dashboard.aggregate()
                    rows = []
                    for node, m in dashboard.metrics.items():
                        rows.append(
                            f"<tr><td>{node}</td><td>{m['energy_kwh']:.3f}</td><td>{m['carbon_g']:.3f}</td><td>{m.get('carbon_intensity',0.0):.3f}</td><td>{m.get('energy_cost',0.0):.3f}</td></tr>"
                        )
                    rows_str = "\n".join(rows)
                    total = agg["total"]
                    sched_rows = "".join(
                        f"<tr><td>{c}</td><td>{s:.3f}</td></tr>" for c, s in dashboard.schedules[-10:]
                    )
                    carbon_saved = sum(s for _, s in dashboard.schedules)
                    html = (
                        "<html><body><h1>Cluster Carbon Dashboard</h1>"
                        "<table border='1'><tr><th>Node</th><th>Energy (kWh)</th><th>Carbon (g)</th><th>Intensity</th><th>Cost</th></tr>"
                        f"{rows_str}</table>"
                        f"<p>Total energy: {total['energy_kwh']:.3f} kWh, Carbon: {total['carbon_g']:.3f} g, Cost: ${total['energy_cost']:.2f}</p>"
                        "<h2>Schedules</h2><table border='1'><tr><th>Cluster</th><th>Carbon Saved</th></tr>"
                        f"{sched_rows}</table>"
                        f"<p>Total carbon saved: {carbon_saved:.3f} g</p>"
                        "</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html.encode())

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        return Handler

    def start(self, host: str = "localhost", port: int = 8090) -> None:
        super().start(host, port)

    def stop(self) -> None:
        super().stop()






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



__all__ = ["MemoryDashboard", "AlignmentDashboard", "IntrospectionDashboard", "InterpretabilityDashboard", "RiskDashboard", "ClusterCarbonDashboard", "MultiAgentDashboard"]

