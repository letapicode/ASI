from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Type


class BaseDashboard:
    """Utility base class to manage an ``HTTPServer`` lifecycle."""

    def __init__(self) -> None:
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

    # --------------------------------------------------------------
    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
        """Return a request handler class bound to this dashboard."""
        raise NotImplementedError

    # --------------------------------------------------------------
    def start(self, host: str = "localhost", port: int = 0) -> None:
        """Start the HTTP server in a background thread."""
        if self.httpd is not None:
            return
        Handler = self.get_handler()
        self.httpd = HTTPServer((host, port), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    # --------------------------------------------------------------
    def stop(self) -> None:
        """Stop the HTTP server."""
        if self.httpd is None:
            return
        assert self.thread is not None
        self.httpd.shutdown()
        self.thread.join(timeout=1.0)
        self.httpd.server_close()
        self.httpd = None
        self.thread = None
        self.port = None


__all__ = ["BaseDashboard"]
