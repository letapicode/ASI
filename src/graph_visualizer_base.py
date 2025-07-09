from __future__ import annotations

import asyncio
import json
import math
import socket
import threading
import time
from typing import Any, Dict, Iterable, List, Tuple

from aiohttp import web


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


def spherical_layout(nodes: Iterable[Dict[str, Any]], radius: float = 3.0) -> Dict[str, Tuple[float, float, float]]:
    """Return a 3D spherical layout."""
    import numpy as np

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

__all__ = [
    "load_graph_json",
    "circular_layout",
    "spherical_layout",
    "WebSocketServer",
]
