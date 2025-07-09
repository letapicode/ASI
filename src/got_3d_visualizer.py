from __future__ import annotations

import io
from typing import Any, Dict, Iterable, List, Tuple

import math
import numpy as np
from aiohttp import web
from ipywidgets import embed
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
try:  # pragma: no cover - prefer package imports
    from asi.graph_visualizer_base import (
        spherical_layout,
        load_graph_json,
        WebSocketServer,
    )
except Exception:  # pragma: no cover - fallback for tests
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        'graph_visualizer_base', Path(__file__).with_name('graph_visualizer_base.py')
    )
    assert spec and spec.loader
    _base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_base)
    spherical_layout = _base.spherical_layout
    load_graph_json = _base.load_graph_json
    WebSocketServer = _base.WebSocketServer


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

    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
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
            sprite = Sprite(SpriteMaterial(map=tex, sizeAttenuation=False), position=[x, y, z + 0.3])
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

    # --------------------------------------------------------------
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


__all__ = ["GOT3DVisualizer", "GOT3DViewer"]
