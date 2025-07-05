from __future__ import annotations

from typing import Iterable, Tuple, List

from .knowledge_graph_memory import KnowledgeGraphMemory, TimedTriple
from .graph_of_thought import GraphOfThought


class TemporalReasoner:
    """Infer ordering of events stored in :class:`KnowledgeGraphMemory`."""

    def __init__(self, kg: KnowledgeGraphMemory) -> None:
        self.kg = kg

    # ------------------------------------------------------------
    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> List[TimedTriple]:
        """Proxy to :meth:`KnowledgeGraphMemory.query_triples`."""
        return self.kg.query_triples(
            subject=subject,
            predicate=predicate,
            object=object,
            start_time=start_time,
            end_time=end_time,
        )

    # ------------------------------------------------------------
    def infer_order(self, triples: Iterable[Tuple[str, str, str]]) -> List[TimedTriple]:
        """Return triples sorted by timestamp, ignoring missing entries."""
        events: List[TimedTriple] = []
        for s, p, o in triples:
            matches = self.kg.query_triples(s, p, o)
            if not matches:
                continue
            # choose earliest timestamp if multiple
            best = min(
                matches,
                key=lambda t: float("inf") if t.timestamp is None else t.timestamp,
            )
            events.append(best)
        events.sort(key=lambda t: float("inf") if t.timestamp is None else t.timestamp)
        return events

    # ------------------------------------------------------------
    def order_nodes_by_time(self, graph: GraphOfThought, nodes: Iterable[int]) -> List[int]:
        """Return ``nodes`` sorted by the timestamps of their ``'triple'`` metadata."""
        nodes = list(nodes)
        if len(nodes) <= 2:
            return nodes
        start, end = nodes[0], nodes[-1]
        middle = nodes[1:-1]
        pairs: List[Tuple[int, float | None]] = []
        for n in middle:
            node = graph.nodes.get(n)
            triple = node.metadata.get("triple") if node is not None else None
            ts: float | None = None
            if triple is not None and len(triple) >= 3:
                res = self.kg.query_triples(*triple[:3])
                if res:
                    ts = res[0].timestamp
            pairs.append((n, ts))
        pairs.sort(key=lambda x: float("inf") if x[1] is None else x[1])
        ordered = [start] + [n for n, _ in pairs] + [end]
        return ordered


__all__ = ["TemporalReasoner"]
