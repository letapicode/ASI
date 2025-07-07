from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, TYPE_CHECKING

try:  # pragma: no cover - optional heavy dep
    import torch
except Exception:  # pragma: no cover - fallback when torch is missing
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .context_summary_memory import ContextSummaryMemory
except Exception:  # pragma: no cover - missing torch or other deps
    ContextSummaryMemory = None  # type: ignore[misc]

try:  # pragma: no cover - optional dependency
    from .analogical_retrieval import analogy_search
except Exception:  # pragma: no cover - fallback if optional modules missing
    def analogy_search(*_args, **_kwargs):  # type: ignore[misc]
        return []
try:  # pragma: no cover - optional dependency
    from .reasoning_history import ReasoningHistoryLogger
except Exception:  # pragma: no cover - fallback for tests
    class ReasoningHistoryLogger:  # type: ignore[dead-code]
        def log(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
            pass

try:  # pragma: no cover - optional dependency
    from .transformer_circuit_analyzer import TransformerCircuitAnalyzer
except Exception:  # pragma: no cover - fallback for tests
    TransformerCircuitAnalyzer = None  # type: ignore[misc]

if TYPE_CHECKING:  # pragma: no cover - for type hints
    from .hierarchical_memory import HierarchicalMemory
import json


@dataclass
class ThoughtNode:
    """Node representing a reasoning step."""

    id: int
    text: str
    metadata: Dict[str, Any] | None = None
    timestamp: float | None = None


class GraphOfThought:
    """Simple searchable reasoning graph."""

    def __init__(
        self,
        analyzer: "TransformerCircuitAnalyzer | None" = None,
        layer: str | None = None,
    ) -> None:
        self.nodes: Dict[int, ThoughtNode] = {}
        self.edges: Dict[int, List[int]] = {}
        self.edge_timestamps: Dict[tuple[int, int], float | None] = {}
        self.analyzer = analyzer
        self.layer = layer

    def add_step(
        self,
        text: str,
        metadata: Dict[str, Any] | None = None,
        node_id: int | None = None,
        sample: "torch.Tensor | None" = None,
        method: str = "gradient",
        timestamp: float | None = None,
    ) -> int:
        """Add a reasoning step and return its node id.

        ``timestamp`` optionally stores when the step occurred.
        """
        if node_id is None:
            node_id = max(self.nodes.keys(), default=-1) + 1
        meta = dict(metadata or {})
        if (
            self.analyzer is not None
            and sample is not None
            and self.layer
            and torch is not None
        ):
            try:
                imps = self.analyzer.head_importance(sample, method=method)
                meta["head_importance"] = imps.tolist()
            except Exception:
                pass
        self.nodes[node_id] = ThoughtNode(node_id, text, meta, timestamp)
        self.edges.setdefault(node_id, [])
        return node_id

    def connect(self, src: int, dst: int, timestamp: float | None = None) -> None:
        """Create a directed edge from ``src`` to ``dst``.

        ``timestamp`` optionally records when the edge was created.
        """
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("unknown node id")
        self.edges.setdefault(src, []).append(dst)
        self.edge_timestamps[(src, dst)] = timestamp

    def summarize_trace(self, trace: Sequence[int]) -> str:
        """Return a natural-language summary of ``trace``."""
        texts = [self.nodes[n].text for n in trace if n in self.nodes]
        return " -> ".join(texts)

    def search(
        self,
        start: int,
        goal_pred: Callable[[ThoughtNode], bool],
        explain: bool = False,
    ) -> List[int] | tuple[List[int], str]:
        """Return path of node ids from ``start`` until ``goal_pred`` is satisfied."""
        if start not in self.nodes:
            raise KeyError("unknown start node")
        visited = {start}
        queue: deque[tuple[int, List[int]]] = deque([(start, [start])])
        while queue:
            node_id, path = queue.popleft()
            node = self.nodes[node_id]
            if goal_pred(node):
                return (path, self.summarize_trace(path)) if explain else path
            for nxt in self.edges.get(node_id, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return ([], "") if explain else []

    def plan_refactor(
        self,
        start: int,
        keyword: str = "refactor",
        explain: bool = False,
        summary_memory: "ContextSummaryMemory | None" = None,
        summary_threshold: int = 10,
    ) -> List[int] | tuple[List[int], Any]:
        """Search for ``keyword`` and optionally summarize the path."""
        key = keyword.lower()
        path, summary = self.search(
            start, lambda node: key in node.text.lower(), explain=True
        )
        if summary_memory is not None and path:
            if len(path) > summary_threshold:
                summary = summary_memory.summarizer.summarize(summary)
                if summary_memory.translator is not None:
                    trans = summary_memory.translator.translate_all(summary)
                    summary = {"summary": summary, "translations": trans}
            return (path, summary)
        return (path, summary) if explain else path

    def self_reflect(self) -> str:
        """Return a concise summary of all reasoning steps."""
        incoming = {dst for dsts in self.edges.values() for dst in dsts}
        starts = [n for n in self.nodes if n not in incoming]
        if not starts:
            starts = sorted(self.nodes)
        visited = set()
        paths = []
        for start in sorted(starts):
            node = start
            texts = []
            while node not in visited:
                visited.add(node)
                texts.append(self.nodes[node].text)
                nexts = self.edges.get(node)
                if not nexts:
                    break
                node = nexts[0]
            if texts:
                paths.append(" -> ".join(texts))
        return "; ".join(paths)

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
                timestamp=node.get("timestamp"),
            )
        for edge in data.get("edges", []):
            if len(edge) == 2:
                src, dst = edge
                ts = None
            else:
                src, dst, ts = edge
            graph.connect(int(src), int(dst), timestamp=ts)
        return graph

    # --------------------------------------------------------------
    def to_json(self) -> dict:
        """Return a JSON-serializable representation of the graph."""
        nodes = [
            {
                "id": n.id,
                "text": n.text,
                "metadata": n.metadata,
                "timestamp": n.timestamp,
            }
            for n in self.nodes.values()
        ]
        edges = [
            [src, dst] + ([] if (ts := self.edge_timestamps.get((src, dst))) is None else [ts])
            for src, dsts in self.edges.items()
            for dst in dsts
        ]
        return {"nodes": nodes, "edges": edges}


class ReasoningDebugger:
    """Detect contradictory steps and loops across one or more reasoning graphs."""

    def __init__(self, graphs: GraphOfThought | Mapping[str, GraphOfThought]) -> None:
        if isinstance(graphs, GraphOfThought):
            self.graphs: Dict[str, GraphOfThought] = {"agent0": graphs}
        else:
            self.graphs = dict(graphs)

    def _loops_in_graph(self, graph: GraphOfThought) -> list[list[int]]:
        loops = []
        for start in graph.nodes:
            path = []
            visited = set()
            node = start
            while node not in visited:
                visited.add(node)
                nexts = graph.edges.get(node, [])
                if not nexts:
                    break
                node = nexts[0]
                path.append(node)
                if node == start:
                    loops.append([start] + path)
                    break
        return loops

    def find_loops(self) -> Dict[str, list[list[int]]]:
        """Return loops detected for each agent graph."""
        result: Dict[str, list[list[int]]] = {}
        for name, graph in self.graphs.items():
            loops = self._loops_in_graph(graph)
            if loops:
                result[name] = loops
        return result

    def find_contradictions(self) -> list[tuple[str, int, str, int]]:
        """Return pairs of nodes whose texts contradict each other."""
        text_map: Dict[str, List[tuple[str, int]]] = {}
        for name, graph in self.graphs.items():
            for node_id, node in graph.nodes.items():
                text_map.setdefault(node.text.lower(), []).append((name, node_id))

        contrad: list[tuple[str, int, str, int]] = []
        for text, entries in text_map.items():
            neg = f"not {text}"
            if neg in text_map:
                for a1, n1 in entries:
                    for a2, n2 in text_map[neg]:
                        if a1 == a2 and n1 == n2:
                            continue
                        pair = (a1, n1, a2, n2)
                        if pair not in contrad:
                            contrad.append(pair)
        return contrad

    def export_graph_data(self) -> Dict[str, list[dict]]:
        """Return nodes and edges formatted for ``GOTVisualizer``."""
        nodes: List[dict] = []
        edges: List[dict] = []
        for agent, graph in self.graphs.items():
            for nid, node in graph.nodes.items():
                nodes.append(
                    {
                        "id": f"{agent}:{nid}",
                        "agent": agent,
                        "orig_id": nid,
                        "text": node.text,
                        "metadata": node.metadata or {},
                    }
                )
            for src, dsts in graph.edges.items():
                for dst in dsts:
                    edges.append(
                        {
                            "source": f"{agent}:{src}",
                            "target": f"{agent}:{dst}",
                        }
                    )
        return {"nodes": nodes, "edges": edges}

    def report(self) -> str:
        """Return a consolidated text report of detected issues."""
        loops = self.find_loops()
        contrad = self.find_contradictions()
        lines: List[str] = []
        for agent, lps in loops.items():
            lines.append(f"Loops for {agent}: {lps}")
        if contrad:
            lines.append("Contradictions:")
            for a1, n1, a2, n2 in contrad:
                lines.append(f"{a1}:{n1} contradicts {a2}:{n2}")
        return "\n".join(lines) if lines else "No issues detected"


class AnalogicalReasoningDebugger:
    """Check reasoning steps against analogy retrieval results."""

    def __init__(
        self,
        graph: GraphOfThought,
        memory: "HierarchicalMemory",
        logger: ReasoningHistoryLogger | None = None,
    ) -> None:
        self.graph = graph
        self.memory = memory
        self.logger = logger or ReasoningHistoryLogger()

    def check_steps(self, k: int = 1) -> List[int]:
        """Return node ids whose expected analogies do not match."""
        mismatches: List[int] = []
        for nid, node in self.graph.nodes.items():
            meta = node.metadata or {}
            if "analogy" not in meta:
                continue
            try:
                query, a, b, expected = meta["analogy"]
            except Exception:
                continue
            _vecs, metas = analogy_search(self.memory, query, a, b, k=k)
            if expected not in metas:
                mismatches.append(nid)
                got = metas[0] if metas else None
                self.logger.log(
                    f"analogy mismatch node {nid}: expected {expected}, got {got}"
                )
        return mismatches

    def report(self, k: int = 1) -> str:
        mism = self.check_steps(k=k)
        return "No analogy mismatches" if not mism else f"Mismatched nodes: {mism}"


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
