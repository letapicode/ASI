from __future__ import annotations

import asyncio
import json
import socket
import threading
from aiohttp import web

try:  # pragma: no cover - prefer package imports
    from asi.graph_of_thought import GraphOfThought
except Exception:  # pragma: no cover - fallback for tests
    from src.graph_of_thought import GraphOfThought  # type: ignore


class ARGOTOverlay:
    """Stream ``GraphOfThought`` data for AR overlays via WebSockets."""

    def __init__(self, graph: GraphOfThought) -> None:
        self.graph = graph
        self.app = web.Application()
        self.app.router.add_get('/ws', self._ws_handler)
        self.clients: list[web.WebSocketResponse] = []
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

    async def _broadcast(self, data: dict) -> None:
        msg = json.dumps(data)
        for ws in list(self.clients):
            try:
                await ws.send_str(msg)
            except Exception:
                self.clients.remove(ws)

    def send_graph(self) -> None:
        """Broadcast the current graph to all connected clients."""
        if self.loop is None:
            return
        data = self.graph.to_json()
        asyncio.run_coroutine_threadsafe(self._broadcast(data), self.loop)

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

    def start(self, host: str = 'localhost', port: int = 8766) -> None:
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
        self.loop = None
        self.runner = None
        self.port = None

__all__ = ['ARGOTOverlay']
