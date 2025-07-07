from __future__ import annotations

import asyncio
import io
import json
import math
import socket
import threading
from typing import Any, Dict, Iterable, List, Tuple

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


class GOT3DVisualizer:
    """Render reasoning graphs in 3D using pythreejs."""

    def __init__(self, nodes: Iterable[Dict[str, Any]], edges: Iterable[Tuple[str, str]]) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)

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
    def _layout(self) -> Dict[str, Tuple[float, float, float]]:
        n = max(len(self.nodes), 1)
        pos: Dict[str, Tuple[float, float, float]] = {}
        for i, node in enumerate(self.nodes):
            phi = math.acos(1 - 2 * (i + 0.5) / n)
            theta = math.pi * (1 + 5 ** 0.5) * (i + 0.5)
            r = 3.0
            pos[str(node["id"])] = (
                r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * math.cos(phi),
            )
        return pos

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
            points: List[float] = []
            for src, dst in self.edges:
                points.extend(pos.get(str(src), (0, 0, 0)))
                points.extend(pos.get(str(dst), (0, 0, 0)))
            arr = np.array(points, dtype="float32").reshape(-1, 3)
            geom = BufferGeometry(attributes={"position": BufferAttribute(arr)})
            line = Line(geometry=geom, material=LineBasicMaterial(color="black"))
            scene.add(line)
        camera = PerspectiveCamera(position=[4, 4, 4], up=[0, 0, 1])
        controls = OrbitControls(controlling=camera)
        renderer = Renderer(scene=scene, camera=camera, controls=[controls], width=600, height=400)
        return renderer

    # --------------------------------------------------------------
    def to_html(self) -> str:
        widget = self.to_widget()
        buf = io.StringIO()
        embed.embed_minimal_html(buf, views=[widget])
        return buf.getvalue()


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
        for ws in list(self.clients):
            try:
                await ws.send_str(html)
            except Exception:
                self.clients.remove(ws)

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
        import time
        time.sleep(0.1)

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
