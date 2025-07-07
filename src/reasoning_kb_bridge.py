from __future__ import annotations

from datetime import datetime
from threading import Event, Thread
from typing import Iterable, List, Sequence

from .graph_of_thought import GraphOfThought
from .knowledge_graph_memory import KnowledgeGraphMemory, TimedTriple
from .reasoning_history import ReasoningHistoryLogger


def graph_to_triples(graph: GraphOfThought) -> List[TimedTriple]:
    """Convert graph nodes and edges into ``TimedTriple`` objects."""
    triples: List[TimedTriple] = []
    for node in graph.nodes.values():
        ts = node.metadata.get("timestamp") if node.metadata else None
        triples.append(TimedTriple(node.text, "node_id", str(node.id), ts))
        if node.metadata:
            for k, v in node.metadata.items():
                if k == "timestamp":
                    continue
                triples.append(TimedTriple(node.text, k, str(v), ts))
    for src, dsts in graph.edges.items():
        src_text = graph.nodes[src].text
        ts = graph.nodes[src].metadata.get("timestamp") if graph.nodes[src].metadata else None
        for dst in dsts:
            dst_text = graph.nodes[dst].text
            triples.append(TimedTriple(src_text, "leads_to", dst_text, ts))
    return triples


class HistoryKGExporter:
    """Periodically export ``ReasoningHistoryLogger`` entries to a knowledge graph."""

    def __init__(self, logger: ReasoningHistoryLogger, kg: KnowledgeGraphMemory, interval: float = 60.0) -> None:
        self.logger = logger
        self.kg = kg
        self.interval = interval
        self._stop = Event()
        self._thread: Thread | None = None
        self._index = 0

    def export_once(self) -> None:
        entries = self.logger.get_history()
        new = entries[self._index :]
        triples: List[TimedTriple] = []
        for ts, summary in new:
            text = summary["summary"] if isinstance(summary, dict) else summary
            try:
                ts_val = datetime.fromisoformat(ts).timestamp()
            except Exception:
                ts_val = None
            triples.append(TimedTriple("reasoning", "summary", str(text), ts_val))
        if triples:
            self.kg.add_triples(triples)
        self._index = len(entries)

    def start(self) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                self.export_once()
                self._stop.wait(self.interval)

        self._thread = Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval)
        self._thread = None
        self._stop.clear()


def get_following_steps(
    kg: KnowledgeGraphMemory,
    step: str,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
) -> List[str]:
    """Return steps that historically followed ``step``."""
    triples = kg.query_triples(
        subject=step,
        predicate="leads_to",
        start_time=start_time,
        end_time=end_time,
    )
    return [t.object for t in triples]


def get_step_metadata(
    kg: KnowledgeGraphMemory,
    step: str,
    key: str,
) -> List[str]:
    """Return metadata values for ``step`` stored under ``key``."""
    triples = kg.query_triples(subject=step, predicate=key)
    return [t.object for t in triples]


__all__ = [
    "graph_to_triples",
    "HistoryKGExporter",
    "get_following_steps",
    "get_step_metadata",
]
