from __future__ import annotations

import asyncio
import io
import json
import math
import socket
import threading
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import time
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
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        nodes = data.get("nodes", [])
        raw_edges = data.get("edges", [])
        if raw_edges and isinstance(raw_edges[0], dict):
            edges = [(e["source"], e["target"]) for e in raw_edges]
        else:
            edges = [(src, dst) for src, dst in raw_edges]
        return cls(nodes, edges)

    # --------------------------------------------------------------
    def _compute_layout(self) -> Dict[str, Tuple[float, float, float]]:
        n = max(len(self.nodes), 1)
        idx = np.arange(len(self.nodes)) + 0.5
        phi = np.arccos(1 - 2 * idx / n)
        theta = np.pi * (1 + math.sqrt(5.0)) * idx
        r = 3.0
        arr = np.stack(
            [
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ],
            axis=1,
        )
        pos = {str(node["id"]): tuple(p) for node, p in zip(self.nodes, arr)}
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


class GOT3DViewer:
    """Serve a 3D graph viewer with optional WebSocket updates."""

    def __init__(self, graph: GOT3DVisualizer) -> None:
        self.graph = graph
        self.app = web.Application()
        self.app.router.add_get("/", self._index)
        self.app.router.add_get("/ws", self._ws_handler)
        self.clients: List[web.WebSocketResponse] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.runner: web.AppRunner | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

    async def _index(self, request: web.Request) -> web.Response:
        html = self.graph.to_html()
        html += (
            "<script>const ws=new WebSocket(`ws://${location.host}/ws`);"
            "ws.onmessage=e=>{document.open();document.write(e.data);document.close();};"
            "</script>"
        )
        return web.Response(text=html, content_type="text/html")

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

    async def _broadcast(self, html: str) -> None:
        if not self.clients:
            return
        results = await asyncio.gather(
            *(ws.send_str(html) for ws in self.clients),
            return_exceptions=True,
        )
        self.clients = [
            ws for ws, res in zip(self.clients, results) if not isinstance(res, Exception)
        ]

    def send_graph(self) -> None:
        if self.loop is None:
            return
        html = self.graph.to_html()
        asyncio.run_coroutine_threadsafe(self._broadcast(html), self.loop)

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

    def start(self, host: str = "localhost", port: int = 8090) -> None:
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


__all__ = ["GOT3DVisualizer", "GOT3DViewer"]
