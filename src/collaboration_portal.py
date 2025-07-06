from __future__ import annotations

import asyncio
import json
import socket
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable, Any

from aiohttp import web

from .telemetry import TelemetryLogger
from .reasoning_history import ReasoningHistoryLogger


class CollaborationPortal:
    """Expose tasks, telemetry metrics and reasoning logs over HTTP."""

    def __init__(
        self,
        tasks: Iterable[str] | None = None,
        telemetry: TelemetryLogger | None = None,
        reasoning: ReasoningHistoryLogger | None = None,
    ) -> None:
        self.tasks = list(tasks or [])
        self.telemetry = telemetry
        self.reasoning = reasoning
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

        # websocket server state
        self.ws_app = web.Application()
        self.ws_app.router.add_get('/ws', self._ws_handler)
        self.ws_clients: list[web.WebSocketResponse] = []
        self.ws_loop: asyncio.AbstractEventLoop | None = None
        self.ws_runner: web.AppRunner | None = None
        self.ws_thread: threading.Thread | None = None
        self.ws_port: int | None = None

        # track timestamps for conflict resolution
        self.task_ts: dict[str, str] = {}
        self.log_ts: str | None = None

    # --------------------------------------------------------------
    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_clients.append(ws)
        # send initial state
        await ws.send_str(json.dumps({"tasks": self.get_tasks(), "logs": self.get_logs()}))
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    self._handle_ws_message(data)
        finally:
            if ws in self.ws_clients:
                self.ws_clients.remove(ws)
        return ws

    def _handle_ws_message(self, data: dict) -> None:
        typ = data.get("type")
        ts = data.get("ts") or datetime.utcnow().isoformat()
        if typ == "add_task" and "task" in data:
            self.add_task(str(data["task"]), ts)
        elif typ == "complete_task" and "task" in data:
            self.complete_task(str(data["task"]), ts)
        elif typ == "log" and "message" in data:
            self.add_log(str(data["message"]), ts)

    async def _broadcast(self, payload: dict) -> None:
        msg = json.dumps(payload)
        for ws in list(self.ws_clients):
            try:
                await ws.send_str(msg)
            except Exception:
                self.ws_clients.remove(ws)

    def _schedule_broadcast(self) -> None:
        if self.ws_loop is None:
            return
        payload = {"tasks": self.get_tasks(), "logs": self.get_logs()}
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.ws_loop)

    # --------------------------------------------------------------
    def add_task(self, task: str, ts: str | None = None) -> None:
        ts = ts or datetime.utcnow().isoformat()
        if ts < self.task_ts.get(task, ""):
            return
        if task not in self.tasks:
            self.tasks.append(task)
        self.task_ts[task] = ts
        self._schedule_broadcast()

    # --------------------------------------------------------------
    def complete_task(self, task: str, ts: str | None = None) -> None:
        ts = ts or datetime.utcnow().isoformat()
        if task in self.tasks and ts >= self.task_ts.get(task, ""):
            self.tasks.remove(task)
            self.task_ts.pop(task, None)
            self._schedule_broadcast()

    # --------------------------------------------------------------
    def get_tasks(self) -> list[str]:
        return list(self.tasks)

    # --------------------------------------------------------------
    def get_metrics(self) -> dict[str, Any]:
        return self.telemetry.get_stats() if self.telemetry else {}

    # --------------------------------------------------------------
    def get_logs(self) -> list[tuple[str, str]]:
        return self.reasoning.get_history() if self.reasoning else []

    # --------------------------------------------------------------
    def add_log(self, message: str, ts: str | None = None) -> None:
        ts = ts or datetime.utcnow().isoformat()
        if self.log_ts is not None and ts <= self.log_ts:
            return
        if self.reasoning is None:
            self.reasoning = ReasoningHistoryLogger()
        self.reasoning.log(message)
        self.log_ts = ts
        self._schedule_broadcast()

    # --------------------------------------------------------------
    def to_html(self) -> str:
        tasks = "".join(f"<li>{t}</li>" for t in self.tasks) or "<li>None</li>"
        logs = (
            "".join(f"<li>{ts}: {msg}</li>" for ts, msg in self.get_logs())
            or "<li>None</li>"
        )
        metrics = self.get_metrics()
        rows = "".join(
            f"<tr><td>{k}</td><td>{v:.3f}</td></tr>" for k, v in metrics.items()
        )
        return (
            "<html><body><h1>Collaboration Portal</h1>"
            "<h2>Active Tasks</h2><ul>" + tasks + "</ul>"
            "<h2>Telemetry</h2><table border='1'>"
            "<tr><th>Metric</th><th>Value</th></tr>" + rows + "</table>"
            "<h2>Reasoning Log</h2><ul>" + logs + "</ul>"
            "</body></html>"
        )

    # --------------------------------------------------------------
    def start(
        self,
        host: str = "localhost",
        port: int = 8070,
        ws_port: int | None = None,
    ) -> None:
        if self.httpd is not None:
            return
        portal = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path == "/tasks":
                    data = json.dumps(portal.get_tasks()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/metrics":
                    data = json.dumps(portal.get_metrics()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/logs":
                    data = json.dumps(portal.get_logs()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    html = portal.to_html().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        self.httpd = HTTPServer((host, port), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

        if ws_port is not None:
            self._start_ws(host, ws_port)

    def _run_ws(self, host: str, port: int) -> None:
        assert self.ws_loop is not None and self.ws_runner is not None
        asyncio.set_event_loop(self.ws_loop)
        self.ws_loop.run_until_complete(self.ws_runner.setup())
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, port))
        _, real_port = sock.getsockname()
        self.ws_port = real_port
        site = web.SockSite(self.ws_runner, sock)
        self.ws_loop.run_until_complete(site.start())
        try:
            self.ws_loop.run_forever()
        finally:
            self.ws_loop.run_until_complete(self.ws_runner.cleanup())

    def _start_ws(self, host: str, port: int) -> None:
        if self.ws_thread is not None:
            return
        self.ws_loop = asyncio.new_event_loop()
        self.ws_runner = web.AppRunner(self.ws_app)
        self.ws_thread = threading.Thread(target=self._run_ws, args=(host, port), daemon=True)
        self.ws_thread.start()
        import time
        time.sleep(0.1)

    # --------------------------------------------------------------
    def stop(self) -> None:
        if self.httpd is not None:
            assert self.thread is not None
            self.httpd.shutdown()
            self.thread.join(timeout=1.0)
            self.httpd.server_close()
            self.httpd = None
            self.thread = None
            self.port = None

        if self.ws_thread is not None and self.ws_loop is not None:
            self.ws_loop.call_soon_threadsafe(self.ws_loop.stop)
            self.ws_thread.join(timeout=1.0)
            self.ws_thread = None
            self.ws_loop = None
            self.ws_runner = None
            self.ws_port = None


__all__ = ["CollaborationPortal"]
