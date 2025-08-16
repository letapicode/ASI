from __future__ import annotations

import asyncio
import io
import json
import math
import socket
import threading
import time
from typing import Any, Dict, Iterable, List, Tuple

import base64
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from aiohttp import web
try:  # pragma: no cover - optional dependency
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    class go:  # type: ignore
        class Figure:
            def __init__(self, data=None) -> None:
                self.data = data

            def update_layout(self, *args, **kwargs) -> None:
                return None

            def to_html(self, **kwargs) -> str:
                return "<html></html>"

        class Scatter:  # type: ignore[dead-code]
            def __init__(self, *args, **kwargs) -> None:
                pass

try:  # pragma: no cover - optional dependency
    from ipywidgets import embed
except Exception:  # pragma: no cover
    class _Embed:
        @staticmethod
        def embed_minimal_html(f, views):
            f.write("<html></html>")

    embed = _Embed()  # type: ignore

try:  # pragma: no cover - optional dependency
    from pythreejs import (
        AmbientLight,
        BufferAttribute,
        BufferGeometry,
        Line,
        LineBasicMaterial,
        Mesh,
        MeshLambertMaterial,
        OrbitControls,
        PerspectiveCamera,
        Renderer,
        Scene,
        SphereGeometry,
        Sprite,
        SpriteMaterial,
        TextTexture,
    )
except Exception:  # pragma: no cover
    class _Stub:  # type: ignore[dead-code]
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _Scene(_Stub):  # type: ignore[dead-code]
        def __init__(self, *args, children=None, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.children = list(children or [])

        def add(self, obj: Any) -> None:
            self.children.append(obj)

    class _Renderer(_Stub):  # type: ignore[dead-code]
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

    AmbientLight = BufferAttribute = BufferGeometry = Line = LineBasicMaterial = Mesh = MeshLambertMaterial = OrbitControls = PerspectiveCamera = SphereGeometry = Sprite = SpriteMaterial = TextTexture = _Stub  # type: ignore
    Renderer = _Renderer  # type: ignore
    Scene = _Scene  # type: ignore


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


# ---------------------------------------------------------------------------
# Helper utilities


def load_graph_json(path: str) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Return nodes and edges loaded from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])
    if raw_edges and isinstance(raw_edges[0], dict):
        edges = [(e["source"], e["target"]) for e in raw_edges]
    else:
        edges = [(src, dst) for src, dst in raw_edges]
    return nodes, edges


def circular_layout(nodes: Iterable[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Return a circular 2D layout mapping node id to coordinates."""
    n = max(len(list(nodes)), 1)
    pos: Dict[str, Tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        pos[str(node["id"])] = (math.cos(angle), math.sin(angle))
    return pos


def spherical_layout(
    nodes: Iterable[Dict[str, Any]], radius: float = 3.0
) -> Dict[str, Tuple[float, float, float]]:
    """Return a 3D spherical layout."""
    nodes_list = list(nodes)
    n = max(len(nodes_list), 1)
    idx = np.arange(len(nodes_list)) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + math.sqrt(5.0)) * idx
    arr = np.stack(
        [
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ],
        axis=1,
    )
    return {str(node["id"]): tuple(p) for node, p in zip(nodes_list, arr)}


class WebSocketServer:
    """Threaded ``aiohttp`` WebSocket server for broadcasting messages."""

    def __init__(self) -> None:
        self.app = web.Application()
        self.app.router.add_get("/ws", self._ws_handler)
        self.clients: List[web.WebSocketResponse] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.runner: web.AppRunner | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.append(ws)
        try:
            async for _ in ws:
                pass
        finally:
            if ws in self.clients:
                self.clients.remove(ws)
        return ws

    async def _broadcast(self, msg: str) -> None:
        if not self.clients:
            return
        results = await asyncio.gather(
            *(ws.send_str(msg) for ws in list(self.clients)),
            return_exceptions=True,
        )
        self.clients = [
            ws for ws, res in zip(list(self.clients), results) if not isinstance(res, Exception)
        ]

    def send(self, msg: str) -> None:
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(msg), self.loop)

    def _run(self, host: str, port: int) -> None:
        assert self.loop is not None
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.runner.setup())
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, port))
        _, real_port = sock.getsockname()
        self.port = real_port
        site = web.SockSite(self.runner, sock)
        self.loop.run_until_complete(site.start())
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(self.runner.cleanup())

    def start(self, host: str = "localhost", port: int = 0) -> None:
        if self.thread is not None:
            return
        self.loop = asyncio.new_event_loop()
        self.runner = web.AppRunner(self.app)
        self.thread = threading.Thread(target=self._run, args=(host, port), daemon=True)
        self.thread.start()
        while self.port is None:
            time.sleep(0.01)

    def stop(self) -> None:
        if self.thread is None or self.loop is None:
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=1.0)
        self.thread = None
        self.runner = None
        self.loop = None
        self.port = None


# ---------------------------------------------------------------------------
# Visualizers


class GOTVisualizer:
    """Render reasoning graphs with Plotly."""

    def __init__(self, nodes: Iterable[Dict[str, Any]], edges: Iterable[Tuple[str, str]]) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)

    @classmethod
    def from_json(cls, path: str) -> "GOTVisualizer":
        nodes, edges = load_graph_json(path)
        return cls(nodes, edges)

    def _layout(self) -> Dict[str, Tuple[float, float]]:
        return circular_layout(self.nodes)

    def to_figure(self) -> go.Figure:
        pos = self._layout()
        edge_x: List[float] = []
        edge_y: List[float] = []
        for src, dst in self.edges:
            x0, y0 = pos.get(str(src), (0, 0))
            x1, y1 = pos.get(str(dst), (0, 0))
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#888", width=1),
            hoverinfo="none",
        )
        node_x = []
        node_y = []
        texts = []
        for node in self.nodes:
            nid = str(node["id"])
            x, y = pos[nid]
            node_x.append(x)
            node_y.append(y)
            texts.append(node.get("text", nid))
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=texts,
            textposition="bottom center",
            marker=dict(size=10, color="#1f77b4"),
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    def to_html(self, title: str = "Reasoning Graph") -> str:
        fig = self.to_figure()
        fig.update_layout(title=title)
        return fig.to_html(include_plotlyjs="cdn")


class GOT3DVisualizer:
    """Render reasoning graphs in 3D using pythreejs."""

    def __init__(self, nodes: Iterable[Dict[str, Any]], edges: Iterable[Tuple[str, str]]) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)
        self._pos: Dict[str, Tuple[float, float, float]] | None = None
        self._edge_arr: np.ndarray | None = None
        self._html: str | None = None

    @classmethod
    def from_json(cls, path: str) -> "GOT3DVisualizer":
        nodes, edges = load_graph_json(path)
        return cls(nodes, edges)

    def _compute_layout(self) -> Dict[str, Tuple[float, float, float]]:
        pos = spherical_layout(self.nodes)
        if self.edges:
            pts = []
            for src, dst in self.edges:
                pts.extend(pos.get(str(src), (0.0, 0.0, 0.0)))
                pts.extend(pos.get(str(dst), (0.0, 0.0, 0.0)))
            self._edge_arr = np.array(pts, dtype="float32").reshape(-1, 3)
        else:
            self._edge_arr = np.zeros((0, 3), dtype="float32")
        return pos

    def _layout(self) -> Dict[str, Tuple[float, float, float]]:
        if self._pos is None:
            self._pos = self._compute_layout()
        return self._pos

    def invalidate(self) -> None:
        self._pos = None
        self._edge_arr = None
        self._html = None

    def to_widget(self) -> Renderer:
        pos = self._layout()
        scene = Scene(children=[AmbientLight(intensity=0.5)])
        for node in self.nodes:
            nid = str(node["id"])
            x, y, z = pos[nid]
            label = str(node.get("text", nid))
            sphere = Mesh(
                geometry=SphereGeometry(radius=0.2, widthSegments=16, heightSegments=16),
                material=MeshLambertMaterial(color="#1f77b4"),
                position=[x, y, z],
            )
            tex = TextTexture(string=label)
            sprite = Sprite(
                SpriteMaterial(map=tex, sizeAttenuation=False), position=[x, y, z + 0.3]
            )
            scene.add(sphere)
            scene.add(sprite)
        if self.edges:
            if self._edge_arr is None:
                self._compute_layout()
            assert self._edge_arr is not None
            geom = BufferGeometry(attributes={"position": BufferAttribute(self._edge_arr)})
            line = Line(geometry=geom, material=LineBasicMaterial(color="black"))
            scene.add(line)
        camera = PerspectiveCamera(position=[4, 4, 4], up=[0, 0, 1])
        controls = OrbitControls(controlling=camera)
        renderer = Renderer(scene=scene, camera=camera, controls=[controls], width=600, height=400)
        return renderer

    def to_html(self) -> str:
        if self._html is None:
            widget = self.to_widget()
            buf = io.StringIO()
            embed.embed_minimal_html(buf, views=[widget])
            self._html = buf.getvalue()
        return self._html


class GOT3DViewer(WebSocketServer):
    """Serve a 3D graph viewer with optional WebSocket updates."""

    def __init__(self, graph: GOT3DVisualizer) -> None:
        super().__init__()
        self.graph = graph
        self.app.router.add_get("/", self._index)

    async def _index(self, request: web.Request) -> web.Response:
        html = self.graph.to_html()
        html += (
            "<script>const ws=new WebSocket(`ws://${location.host}/ws`);"
            "ws.onmessage=e=>{document.open();document.write(e.data);document.close();};"
            "</script>"
        )
        return web.Response(text=html, content_type="text/html")

    def send_graph(self) -> None:
        html = self.graph.to_html()
        self.send(html)


# ---------------------------------------------------------------------------
# D3-based visualizers


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
__all__ = [
    "load_graph_json",
    "circular_layout",
    "spherical_layout",
    "WebSocketServer",
    "GOTVisualizer",
    "GOT3DVisualizer",
    "GOT3DViewer",
    "D3GraphVisualizer",
    "LineageVisualizer",
    "DatasetLineageDashboard",
    "KGVisualizer",
]
