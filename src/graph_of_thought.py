from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence
import json


@dataclass
class ThoughtNode:
    """Node representing a reasoning step."""

    id: int
    text: str
    metadata: Dict[str, Any] | None = None


class GraphOfThought:
    """Simple searchable reasoning graph."""

    def __init__(self) -> None:
        self.nodes: Dict[int, ThoughtNode] = {}
        self.edges: Dict[int, List[int]] = {}

    def add_step(
        self, text: str, metadata: Dict[str, Any] | None = None, node_id: int | None = None
    ) -> int:
        """Add a reasoning step and return its node id."""
        if node_id is None:
            node_id = max(self.nodes.keys(), default=-1) + 1
        self.nodes[node_id] = ThoughtNode(node_id, text, metadata or {})
        self.edges.setdefault(node_id, [])
        return node_id

    def connect(self, src: int, dst: int) -> None:
        """Create a directed edge from ``src`` to ``dst``."""
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("unknown node id")
        self.edges.setdefault(src, []).append(dst)

    def search(self, start: int, goal_pred: Callable[[ThoughtNode], bool]) -> List[int]:
        """Return path of node ids from ``start`` until ``goal_pred`` is satisfied."""
        if start not in self.nodes:
            raise KeyError("unknown start node")
        visited = {start}
        queue: deque[tuple[int, List[int]]] = deque([(start, [start])])
        while queue:
            node_id, path = queue.popleft()
            node = self.nodes[node_id]
            if goal_pred(node):
                return path
            for nxt in self.edges.get(node_id, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return []

    def plan_refactor(self, start: int, keyword: str = "refactor") -> List[int]:
        """Search for a node containing ``keyword`` in its text."""
        key = keyword.lower()
        return self.search(start, lambda node: key in node.text.lower())

    @classmethod
    def from_json(cls, path: str) -> "GraphOfThought":
        """Load graph from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        graph = cls()
        for node in data.get("nodes", []):
            graph.add_step(
                node.get("text", ""),
                metadata=node.get("metadata"),
                node_id=int(node["id"]),
            )
        for src, dst in data.get("edges", []):
            graph.connect(int(src), int(dst))
        return graph


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for planning code refactors."""
    import argparse

    parser = argparse.ArgumentParser(description="Search reasoning graph for refactor plan")
    parser.add_argument("graph", help="Path to graph JSON file")
    parser.add_argument("start", type=int, help="Start node id")
    parser.add_argument("--goal", default="refactor", help="Goal keyword")
    args = parser.parse_args(argv)

    graph = GraphOfThought.from_json(args.graph)
    path = graph.plan_refactor(args.start, args.goal)
    if path:
        print(" -> ".join(str(n) for n in path))
    else:
        print("No plan found")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
