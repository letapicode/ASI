from __future__ import annotations

"""WebSocket-based collaborative editor for ``GraphOfThought``."""

import asyncio
import json
from typing import Any, Dict, Set, Optional

from .graph_of_thought import GraphOfThought
from .nl_graph_editor import NLGraphEditor

try:  # pragma: no cover - optional dependency
    import websockets
except Exception:  # pragma: no cover
    websockets = None  # type: ignore


class CollaborativeGoTEditor:
    """Serve a reasoning graph to multiple editors with simple CRDT merge."""

    def __init__(self, graph: GraphOfThought) -> None:
        self.graph = graph
        self.editor = NLGraphEditor(graph)
        self.clients: Set[Any] = set()
        self.version = 0
        self.lock = asyncio.Lock()

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        if not self.clients:
            return
        msg = json.dumps(payload)
        await asyncio.gather(*[c.send(msg) for c in list(self.clients)])

    async def _handler(self, websocket: Any) -> None:
        self.clients.add(websocket)
        try:
            await websocket.send(
                json.dumps({"graph": self.graph.to_json(), "version": self.version})
            )
            async for raw in websocket:
                data = json.loads(raw)
                cmd = data.get("cmd", "")
                lang = data.get("lang")
                async with self.lock:
                    before = set(self.graph.nodes)
                    result = self.editor.apply(cmd)
                    if lang and hasattr(self.graph, "translate_node"):
                        new_ids = set(self.graph.nodes) - before
                        for nid in new_ids:
                            node = self.graph.nodes[nid]
                            node.metadata = dict(node.metadata or {})
                            node.metadata.setdefault("lang", lang)
                    self.version += 1
                    await self._broadcast({"result": result, "version": self.version})
        finally:
            self.clients.discard(websocket)

    def start(self, host: str = "localhost", port: int = 8765) -> asyncio.AbstractServer:
        if websockets is None:
            raise ImportError("websockets package is required")
        return websockets.serve(self._handler, host, port)


__all__ = ["CollaborativeGoTEditor"]
