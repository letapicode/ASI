from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable, Any
import asyncio
import socket
from aiohttp import web
from urllib.parse import urlparse, parse_qs

from .telemetry import TelemetryLogger
from .reasoning_history import ReasoningHistoryLogger
from .data_ingest import CrossLingualTranslator


class CollaborationPortal:
    """Expose tasks, telemetry metrics and reasoning logs over HTTP."""

    def __init__(
        self,
        tasks: Iterable[str] | None = None,
        telemetry: TelemetryLogger | None = None,
        reasoning: ReasoningHistoryLogger | None = None,
        translator: CrossLingualTranslator | None = None,
    ) -> None:
        self.tasks = list(tasks or [])
        self.telemetry = telemetry
        self.reasoning = reasoning
        self.translator = translator
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None
        self.ws_app: web.Application | None = None
        self.ws_loop: asyncio.AbstractEventLoop | None = None
        self.ws_runner: web.AppRunner | None = None
        self.ws_thread: threading.Thread | None = None
        self.ws_port: int | None = None
        self.ws_clients: list[web.WebSocketResponse] = []
        self._task_ts: float = 0.0
        self._log_ts: float = 0.0

    # --------------------------------------------------------------
    def add_task(self, task: str) -> None:
        self.tasks.append(task)
        self._task_ts = asyncio.get_event_loop().time()
        self._broadcast_state()

    # --------------------------------------------------------------
    def complete_task(self, task: str) -> None:
        if task in self.tasks:
            self.tasks.remove(task)
            self._task_ts = asyncio.get_event_loop().time()
            self._broadcast_state()

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
    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_clients.append(ws)
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    ts = data.get("timestamp", 0)
                    if "tasks" in data and ts > self._task_ts:
                        self.tasks = data["tasks"]
                        self._task_ts = ts
                        await self._broadcast_state()
                    if "logs" in data and ts > self._log_ts:
                        self.reasoning.entries = data["logs"]
                        self._log_ts = ts
                        await self._broadcast_state()
        finally:
            if ws in self.ws_clients:
                self.ws_clients.remove(ws)
        return ws

    async def _broadcast_state(self) -> None:
        if not self.ws_clients:
            return
        data = {
            "tasks": self.tasks,
            "logs": self.get_logs(),
            "timestamp": max(self._task_ts, self._log_ts),
        }
        msg = json.dumps(data)
        for ws in list(self.ws_clients):
            try:
                await ws.send_str(msg)
            except Exception:
                self.ws_clients.remove(ws)

    def _broadcast_state_sync(self) -> None:
        if self.ws_loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_state(), self.ws_loop)

    def broadcast(self) -> None:
        """Broadcast current state over WebSocket."""
        self._broadcast_state_sync()

    # --------------------------------------------------------------
    def start(self, host: str = "localhost", port: int = 8070) -> None:
        if self.httpd is not None:
            return
        portal = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                parsed = urlparse(self.path)
                lang = None
                params = parse_qs(parsed.query)
                if "lang" in params:
                    lang = params["lang"][0]
                if lang is None:
                    header = self.headers.get("Accept-Language")
                    if header:
                        lang = header.split(",")[0].strip()

                if parsed.path == "/tasks":
                    tasks = portal.get_tasks()
                    if portal.translator is not None:
                        if lang:
                            tasks = [portal.translator.translate(t, lang) for t in tasks]
                        else:
                            tasks = [
                                {"task": t, "translations": portal.translator.translate_all(t)}
                                for t in tasks
                            ]
                    data = json.dumps(tasks).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif parsed.path == "/metrics":
                    metrics = portal.get_metrics()
                    if portal.translator is not None:
                        if lang:
                            metrics = {
                                portal.translator.translate(k, lang): v
                                for k, v in metrics.items()
                            }
                        else:
                            metrics = {
                                k: {
                                    "value": v,
                                    "translations": portal.translator.translate_all(k),
                                }
                                for k, v in metrics.items()
                            }
                    data = json.dumps(metrics).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif parsed.path == "/logs":
                    logs = []
                    for ts, msg in portal.get_logs():
                        if portal.translator is not None:
                            text = msg.get("summary", msg) if isinstance(msg, dict) else msg
                            translations = msg.get("translations") if isinstance(msg, dict) else None
                            if lang:
                                if translations and lang in translations:
                                    msg_out = translations[lang]
                                else:
                                    msg_out = portal.translator.translate(text, lang)
                            else:
                                msg_out = translations if translations else portal.translator.translate_all(text)
                            logs.append((ts, msg_out))
                        else:
                            logs.append((ts, msg))
                    data = json.dumps(logs).encode()
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

    # --------------------------------------------------------------
    def start_ws(self, host: str = "localhost", port: int = 8767) -> None:
        if self.ws_thread is not None:
            return
        self.ws_loop = asyncio.new_event_loop()
        self.ws_app = web.Application()
        self.ws_app.router.add_get('/ws', self._ws_handler)
        self.ws_runner = web.AppRunner(self.ws_app)

        def _run() -> None:
            assert self.ws_loop and self.ws_runner
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

        self.ws_thread = threading.Thread(target=_run, daemon=True)
        self.ws_thread.start()
        import time
        time.sleep(0.1)

    # --------------------------------------------------------------
    def stop_ws(self) -> None:
        if self.ws_thread is None or self.ws_loop is None:
            return
        self.ws_loop.call_soon_threadsafe(self.ws_loop.stop)
        self.ws_thread.join(timeout=1.0)
        self.ws_thread = None
        self.ws_loop = None
        self.ws_runner = None
        self.ws_app = None
        self.ws_port = None
        self.ws_clients.clear()

    # --------------------------------------------------------------
    def stop(self) -> None:
        if self.httpd is None:
            return
        assert self.thread is not None
        self.httpd.shutdown()
        self.thread.join(timeout=1.0)
        self.httpd.server_close()
        self.httpd = None
        self.thread = None
        self.port = None
        self.stop_ws()


__all__ = ["CollaborationPortal"]
