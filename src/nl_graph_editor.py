from __future__ import annotations

import re
from typing import Any, Dict

from .graph_of_thought import GraphOfThought


class NLGraphEditor:
    """Apply simple natural-language edit commands to a ``GraphOfThought``."""

    def __init__(self, graph: GraphOfThought) -> None:
        self.graph = graph

    # --------------------------------------------------------------
    def _find_node(self, token: str) -> int:
        token = token.strip()
        if token.isdigit() and int(token) in self.graph.nodes:
            return int(token)
        for nid, node in self.graph.nodes.items():
            if node.text.lower() == token.lower():
                return nid
        raise KeyError(f"unknown node {token}")

    # --------------------------------------------------------------
    def apply(self, command: str) -> Dict[str, Any]:
        cmd = command.lower().strip()
        if m := re.search(r"add edge from (.+?) to (.+)", cmd):
            src = self._find_node(m.group(1))
            dst = self._find_node(m.group(2))
            self.graph.connect(src, dst)
            return {"status": "ok", "action": "add_edge", "src": src, "dst": dst}
        if m := re.search(r"remove edge from (.+?) to (.+)", cmd):
            src = self._find_node(m.group(1))
            dst = self._find_node(m.group(2))
            if src in self.graph.edges:
                self.graph.edges[src] = [d for d in self.graph.edges[src] if d != dst]
            return {"status": "ok", "action": "remove_edge", "src": src, "dst": dst}
        if m := re.search(r"merge nodes (.+?) and (.+)", cmd):
            a = self._find_node(m.group(1))
            b = self._find_node(m.group(2))
            new_text = f"{self.graph.nodes[a].text} {self.graph.nodes[b].text}"
            new_id = self.graph.add_step(new_text)
            outgoing = list({*self.graph.edges.get(a, []), *self.graph.edges.get(b, [])})
            for tgt in outgoing:
                self.graph.connect(new_id, tgt)
            for src, dsts in list(self.graph.edges.items()):
                self.graph.edges[src] = [new_id if d in (a, b) else d for d in dsts]
            for nid in (a, b):
                self.graph.nodes.pop(nid, None)
                self.graph.edges.pop(nid, None)
            return {"status": "ok", "action": "merge", "new_id": new_id}
        if m := re.search(r"add node (.+)", cmd):
            text = command[ m.start(1) : ].strip() if m.lastindex else command
            node_id = self.graph.add_step(text)
            return {"status": "ok", "action": "add_node", "id": node_id}
        if m := re.search(r"remove node (.+)", cmd):
            nid = self._find_node(m.group(1))
            self.graph.nodes.pop(nid, None)
            self.graph.edges.pop(nid, None)
            for src, dsts in list(self.graph.edges.items()):
                self.graph.edges[src] = [d for d in dsts if d != nid]
            return {"status": "ok", "action": "remove_node", "id": nid}
        raise ValueError("unrecognized command")


__all__ = ["NLGraphEditor"]
