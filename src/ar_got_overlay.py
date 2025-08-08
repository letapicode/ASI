from __future__ import annotations

import json
from aiohttp import web
try:  # pragma: no cover - prefer package imports
    from asi.graph_visualizer import WebSocketServer
except Exception:  # pragma: no cover - fallback for tests
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        'graph_visualizer', Path(__file__).with_name('graph_visualizer.py')
    )
    assert spec and spec.loader
    _base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_base)
    WebSocketServer = _base.WebSocketServer

try:  # pragma: no cover - prefer package imports
    from asi.graph_of_thought import GraphOfThought
except Exception:  # pragma: no cover - fallback for tests
    from src.graph_of_thought import GraphOfThought  # type: ignore


class ARGOTOverlay(WebSocketServer):
    """Stream ``GraphOfThought`` data for AR overlays via WebSockets."""

    def __init__(self, graph: GraphOfThought) -> None:
        super().__init__()
        self.graph = graph

    def send_graph(self) -> None:
        """Broadcast the current graph to all connected clients."""
        data = self.graph.to_json()
        self.send(json.dumps(data))

__all__ = ['ARGOTOverlay']
