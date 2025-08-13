from __future__ import annotations

import base64
import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable

from .dataset_lineage import DatasetLineageManager, LineageStep
from .knowledge_graph_memory import KnowledgeGraphMemory

try:  # pragma: no cover - fallback when not packaged
    from .dashboard_import_helper import load_base_dashboard
except Exception:  # pragma: no cover
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "dashboard_import_helper", Path(__file__).with_name("dashboard_import_helper.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    sys.modules.setdefault("dashboard_import_helper", module)
    load_base_dashboard = module.load_base_dashboard  # type: ignore


BaseDashboard = load_base_dashboard(__file__)


class D3GraphVisualizer:
    """Serve ``graph_json`` over HTTP and render D3 visualizations."""

    def __init__(self, graph_endpoint: str) -> None:
        self.graph_endpoint = graph_endpoint
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

    # ------------------------------------------------------------------
    def graph_json(self) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------
    def to_html(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------
    def start(self, port: int = 8000) -> None:
        if self.httpd is not None:
            return

        visualizer = self
        endpoint = self.graph_endpoint

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path == endpoint:
                    data = json.dumps(visualizer.graph_json()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    html = visualizer.to_html().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        self.httpd = HTTPServer(("localhost", port), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if self.httpd is None:
            return
        assert self.thread is not None
        self.httpd.shutdown()
        self.thread.join(timeout=1)
        self.httpd.server_close()
        self.httpd = None
        self.thread = None
        self.port = None


def _build_graph(steps: Iterable[LineageStep]) -> Dict[str, list]:
    nodes: Dict[str, Dict[str, str]] = {}
    links: list[Dict[str, str]] = []
    for step in steps:
        for inp in step.inputs:
            nodes[inp] = {"id": inp}
        for out in step.outputs.keys():
            nodes[out] = {"id": out}
            for inp in step.inputs:
                links.append({"source": inp, "target": out, "note": step.note})
    return {"nodes": list(nodes.values()), "links": links}


class LineageVisualizer(D3GraphVisualizer):
    """Serve an interactive graph of dataset lineage."""

    def __init__(self, manager: DatasetLineageManager) -> None:
        super().__init__("/graph")
        self.manager = manager
        self._fairness_img: str | None = None

    # --------------------------------------------------------------
    def graph_json(self) -> Dict[str, list]:  # type: ignore[override]
        return _build_graph(self.manager.steps)

    # --------------------------------------------------------------
    def _load_fairness(self) -> None:
        if self._fairness_img is not None:
            return
        fname = f"{Path(self.manager.root).stem}_fairness.png"
        path = Path("docs/datasets") / fname
        if path.exists():
            b64 = base64.b64encode(path.read_bytes()).decode()
            self._fairness_img = f"data:image/png;base64,{b64}"
        else:
            self._fairness_img = ""

    # --------------------------------------------------------------
    def to_html(self) -> str:  # type: ignore[override]
        self._load_fairness()
        img = f"<img src='{self._fairness_img}'>" if self._fairness_img else ""
        ep = self.graph_endpoint
        return (
            "<html><body><h1>Dataset Lineage</h1>" + img +
            "<script src='https://cdn.jsdelivr.net/npm/d3@7'></script>"
            "<svg width='800' height='600'></svg>"
            "<script>"
            f"fetch('{ep}').then(r=>r.json()).then(data=>{{"
            "const svg=d3.select('svg');"
            "const width=+svg.attr('width'),height=+svg.attr('height');"
            "const sim=d3.forceSimulation(data.nodes)"
            ".force('link',d3.forceLink(data.links).id(d=>d.id).distance(40))"
            ".force('charge',d3.forceManyBody().strength(-200))"
            ".force('center',d3.forceCenter(width/2,height/2));"
            "const link=svg.append('g').selectAll('line')"
            ".data(data.links).enter().append('line').attr('stroke','#999');"
            "const node=svg.append('g').selectAll('circle')"
            ".data(data.nodes).enter().append('circle').attr('r',5)"
            ".call(d3.drag()"
            ".on('start',e=>{if(!e.active)sim.alphaTarget(0.3).restart();e.subject.fx=e.subject.x;e.subject.fy=e.subject.y;})"
            ".on('drag',e=>{e.subject.fx=e.x;e.subject.fy=e.y;})"
            ".on('end',e=>{if(!e.active)sim.alphaTarget(0);e.subject.fx=null;e.subject.fy=null;}));"
            "const text=svg.append('g').selectAll('text')"
            ".data(data.nodes).enter().append('text').text(d=>d.id.split('/').pop());"
            "sim.on('tick',()=>{"
            "link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)"
            ".attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);"
            "node.attr('cx',d=>d.x).attr('cy',d=>d.y);"
            "text.attr('x',d=>d.x+6).attr('y',d=>d.y+4);"
            "});"
            "});"
            "</script></body></html>"
        )


class DatasetLineageDashboard(BaseDashboard):
    """Serve dataset lineage graphs with basic filtering and search."""

    def __init__(self, manager: DatasetLineageManager) -> None:
        super().__init__()
        self.manager = manager

    # --------------------------------------------------------------
    def _filter_steps(
        self,
        *,
        query: str | None = None,
        note: str | None = None,
        inp: str | None = None,
        out: str | None = None,
    ) -> list[LineageStep]:
        steps = self.manager.steps
        results: list[LineageStep] = []
        for s in steps:
            if note and note not in s.note:
                continue
            if inp and inp not in s.inputs:
                continue
            if out and out not in s.outputs:
                continue
            if query:
                vals = [str(v) for v in s.outputs.values()]
                hay = " ".join([s.note] + s.inputs + list(s.outputs.keys()) + vals)
                if query not in hay:
                    continue
            results.append(s)
        return results

    # --------------------------------------------------------------
    def graph_json(self, steps: Iterable[LineageStep] | None = None) -> Dict[str, list]:
        return _build_graph(self.manager.steps if steps is None else steps)

    # --------------------------------------------------------------
    def steps_json(self, steps: Iterable[LineageStep]) -> list[Dict[str, Any]]:
        return [asdict(s) for s in steps]

    # --------------------------------------------------------------
    def to_html(self) -> str:
        return (
            "<html><body><h1>Dataset Lineage Dashboard</h1>"
            "<input id='q' placeholder='search'><button onclick='load()'>Search</button>"
            "<script src='https://cdn.jsdelivr.net/npm/d3@7'></script>"
            "<svg width='800' height='600'></svg>"
            "<script>"
            "function draw(data){const svg=d3.select('svg');svg.selectAll('*').remove();"
            "const width=+svg.attr('width'),height=+svg.attr('height');"
            "const sim=d3.forceSimulation(data.nodes)"
            ".force('link',d3.forceLink(data.links).id(d=>d.id).distance(40))"
            ".force('charge',d3.forceManyBody().strength(-200))"
            ".force('center',d3.forceCenter(width/2,height/2));"
            "const link=svg.append('g').selectAll('line')"
            ".data(data.links).enter().append('line').attr('stroke','#999');"
            "const node=svg.append('g').selectAll('circle')"
            ".data(data.nodes).enter().append('circle').attr('r',5);"
            "const text=svg.append('g').selectAll('text')"
            ".data(data.nodes).enter().append('text').text(d=>d.id.split('/').pop());"
            "sim.on('tick',()=>{"
            "link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)"
            ".attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);"
            "node.attr('cx',d=>d.x).attr('cy',d=>d.y);"
            "text.attr('x',d=>d.x+6).attr('y',d=>d.y+4);"
            "});"
            "}"
            "function load(){const q=document.getElementById('q').value;fetch('/steps?q='+q).then(r=>r.json()).then(steps=>{draw({nodes:[],links:[]});fetch('/graph').then(r=>r.json()).then(g=>{draw(g);});});}"
            "load();"
            "</script></body></html>"
        )

    # --------------------------------------------------------------
    def start(self, port: int = 8000) -> None:
        if getattr(self, "httpd", None) is not None:  # type: ignore[attr-defined]
            return

        dashboard = self

        from urllib.parse import parse_qs, urlparse

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                parsed = urlparse(self.path)
                if parsed.path == "/graph":
                    data = json.dumps(dashboard.graph_json()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif parsed.path == "/steps":
                    qs = parse_qs(parsed.query)
                    steps = dashboard._filter_steps(
                        query=qs.get("q", [None])[0],
                        note=qs.get("note", [None])[0],
                        inp=qs.get("inp", [None])[0],
                        out=qs.get("out", [None])[0],
                    )
                    data = json.dumps(dashboard.steps_json(steps)).encode()
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

        self.httpd = HTTPServer(("localhost", port), Handler)  # type: ignore[attr-defined]
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)  # type: ignore[attr-defined]
        self.thread.start()

    # --------------------------------------------------------------
    def stop(self) -> None:
        if getattr(self, "httpd", None) is None:  # type: ignore[attr-defined]
            return
        assert self.thread is not None
        self.httpd.shutdown()  # type: ignore[attr-defined]
        self.thread.join(timeout=1)
        self.httpd.server_close()  # type: ignore[attr-defined]
        self.httpd = None  # type: ignore[attr-defined]
        self.thread = None
        self.port = None


class KGVisualizer(D3GraphVisualizer):
    """Serve an interactive view of a ``KnowledgeGraphMemory``."""

    def __init__(self, kg: KnowledgeGraphMemory) -> None:
        super().__init__("/kg/graph")
        self.kg = kg

    # --------------------------------------------------------------
    def graph_json(self) -> Dict[str, Any]:  # type: ignore[override]
        nodes: Dict[str, Dict[str, str]] = {}
        edges: list[Dict[str, Any]] = []
        for (s, p, o), ts in zip(self.kg.triples, self.kg.timestamps):
            nodes.setdefault(s, {"id": s})
            nodes.setdefault(o, {"id": o})
            edges.append({
                "source": s,
                "target": o,
                "predicate": p,
                "timestamp": ts,
            })
        return {"nodes": list(nodes.values()), "edges": edges}

    # --------------------------------------------------------------
    def to_html(self) -> str:  # type: ignore[override]
        ep = self.graph_endpoint
        return (
            "<html><body><h1>Knowledge Graph</h1>"
            "<script src='https://cdn.jsdelivr.net/npm/d3@7'></script>"
            "<svg width='800' height='600'></svg>"
            "<script>"
            f"fetch('{ep}').then(r=>r.json()).then(data=>{{"
            "const svg=d3.select('svg');"
            "const w=+svg.attr('width'),h=+svg.attr('height');"
            "const sim=d3.forceSimulation(data.nodes)"
            ".force('link',d3.forceLink(data.edges).id(d=>d.id).distance(60))"
            ".force('charge',d3.forceManyBody().strength(-200))"
            ".force('center',d3.forceCenter(w/2,h/2));"
            "const link=svg.append('g').selectAll('line')"
            ".data(data.edges).enter().append('line').attr('stroke','#999');"
            "const node=svg.append('g').selectAll('circle')"
            ".data(data.nodes).enter().append('circle').attr('r',5).attr('fill','#69b');"
            "node.append('title').text(d=>d.id);"
            "const label=svg.append('g').selectAll('text')"
            ".data(data.edges).enter().append('text').text(d=>d.predicate+(d.timestamp?' @'+d.timestamp:''));"
            "sim.on('tick',()=>{"
            "link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)"
            ".attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);"
            "node.attr('cx',d=>d.x).attr('cy',d=>d.y);"
            "label.attr('x',d=>(d.source.x+d.target.x)/2)"
            ".attr('y',d=>(d.source.y+d.target.y)/2);"
            "});"
            "});"
            "</script></body></html>"
        )


__all__ = ["LineageVisualizer", "DatasetLineageDashboard", "KGVisualizer", "D3GraphVisualizer"]
