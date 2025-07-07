from __future__ import annotations

from typing import Iterable, Tuple, List

from .knowledge_graph_memory import KnowledgeGraphMemory, TimedTriple
from .graph_of_thought import GraphOfThought


class TemporalReasoner:
    """Infer ordering of events stored in :class:`KnowledgeGraphMemory`."""

    def __init__(self, kg: KnowledgeGraphMemory) -> None:
        self.kg = kg
        self._time_cache: dict[tuple[str, str, str], float | None] = {}

    # ------------------------------------------------------------
    def _timestamp(self, triple: tuple[str, str, str]) -> float | None:
        """Return cached timestamp for ``triple``."""
        key = triple[:3]
        if key in self._time_cache:
            return self._time_cache[key]
        res = self.kg.query_triples(*key)
        if res:
            ts = min(
                (t.timestamp for t in res if t.timestamp is not None),
                default=None,
            )
        else:
            ts = None
        self._time_cache[key] = ts
        return ts

    def _timestamp_of_node(
        self, graph: GraphOfThought, prev: int, node_id: int
    ) -> float | None:
        node = graph.nodes.get(node_id)
        ts = None
        if node is not None:
            if node.timestamp is not None:
                ts = node.timestamp
            else:
                triple = node.metadata.get("triple") if node.metadata else None
                if triple is not None and len(triple) >= 3:
                    ts = self._timestamp(tuple(triple[:3]))
        if ts is None:
            ts = graph.edge_timestamps.get((prev, node_id))
        return ts

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
            ts = self._timestamp((s, p, o))
            if ts is None:
                continue
            matches = [TimedTriple(s, p, o, ts)]
            best = matches[0]
            events.append(best)
        events.sort(key=lambda t: float("inf") if t.timestamp is None else t.timestamp)
        return events

    # ------------------------------------------------------------
    def order_nodes_by_time(
        self,
        graph: GraphOfThought,
        nodes: Iterable[int],
        *,
        compress: bool = False,
    ) -> List[int]:
        """Return ``nodes`` ordered by timestamp.

        If ``compress`` is ``True`` consecutive nodes sharing the same timestamp
        are collapsed to a single node (keeping the first occurrence).
        """

        nodes = list(nodes)
        if len(nodes) <= 2:
            return nodes
        start, end = nodes[0], nodes[-1]
        middle = nodes[1:-1]
        pairs: List[Tuple[int, float | None]] = []
        prev = start
        for n in middle:
            ts = self._timestamp_of_node(graph, prev, n)
            pairs.append((n, ts))
            prev = n

        pairs.sort(key=lambda x: float("inf") if x[1] is None else x[1])

        if compress:
            pairs = self._compress_pairs(pairs)

        ordered = [start] + [n for n, _ in pairs] + [end]
        return ordered

    @staticmethod
    def _compress_pairs(pairs: List[Tuple[int, float | None]]) -> List[Tuple[int, float | None]]:
        compressed: List[Tuple[int, float | None]] = []
        last_ts = object()
        for n, ts in pairs:
            if ts == last_ts:
                continue
            compressed.append((n, ts))
            last_ts = ts
        return compressed


__all__ = ["TemporalReasoner"]
