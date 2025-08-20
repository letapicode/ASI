from __future__ import annotations

import time
from typing import Any, Sequence, List
import numpy as np

try:  # optional torch dependency
    import torch
except Exception:  # pragma: no cover - allow running without torch
    from .torch_fallback import torch

from .graph_of_thought import GraphOfThought

try:  # optional dependency
    from .context_summary_memory import ContextSummaryMemory
except Exception:  # pragma: no cover - stubs for tests
    ContextSummaryMemory = None  # type: ignore[misc]

try:
    from .telemetry import TelemetryLogger
except Exception:  # pragma: no cover - fallback
    class TelemetryLogger:  # type: ignore[dead-code]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.events = []


def _log_events(
    telemetry: TelemetryLogger | None, event: str, items: Sequence[Any], key: str
) -> None:
    """Record pruning ``items`` in ``telemetry`` if provided."""
    if telemetry is None:
        return
    for item in items:
        telemetry.events.append({"event": event, key: item})


class GraphPruningManager:
    """Monitor a reasoning graph and prune outdated or low-degree nodes."""

    def __init__(
        self,
        *,
        degree_threshold: int = 1,
        age_threshold: float | None = None,
        memory: ContextSummaryMemory | None = None,
        summarizer: Any | None = None,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self.degree_threshold = int(degree_threshold)
        self.age_threshold = age_threshold
        self.memory = memory
        self.summarizer = summarizer or (memory.summarizer if memory is not None else None)
        self.telemetry = telemetry
        self.graph: GraphOfThought | None = None

    def _incoming_counts(self) -> dict[int, int]:
        assert self.graph is not None
        counts: dict[int, int] = {}
        for dsts in self.graph.edges.values():
            for d in dsts:
                counts[d] = counts.get(d, 0) + 1
        return counts

    def attach(self, graph: GraphOfThought) -> None:
        """Attach to a :class:`GraphOfThought` instance."""
        self.graph = graph

    def _summarize(self, nodes: Sequence[int]) -> None:
        if not nodes or self.memory is None or self.summarizer is None or self.graph is None:
            return
        text = "; ".join(self.graph.nodes[n].text for n in nodes if n in self.graph.nodes)
        summary = self.summarizer.summarize(text)
        info = {"summary": summary}
        if self.memory.translator is not None:
            info["translations"] = self.memory.translator.translate_all(summary)
        vec = self.summarizer.expand(summary)
        comp = self.memory.compressor.encoder(vec.unsqueeze(0))
        self.memory.add_compressed(comp, [{"ctxsum": info}])

    def prune_low_degree(self, threshold: int | None = None) -> Sequence[int]:
        if self.graph is None:
            return []
        deg = self.degree_threshold if threshold is None else threshold
        incoming = self._incoming_counts()
        remove = [
            n
            for n in list(self.graph.nodes)
            if len(self.graph.edges.get(n, [])) + incoming.get(n, 0) <= deg
        ]
        if remove:
            self._summarize(remove)
            self._remove_nodes(remove)
        return remove

    def prune_old_nodes(self, *, age: float | None = None, now: float | None = None) -> Sequence[int]:
        if self.graph is None:
            return []
        if age is None:
            age = self.age_threshold
        if age is None:
            return []
        cur = time.time() if now is None else now
        remove = [
            n
            for n, node in list(self.graph.nodes.items())
            if node.timestamp is not None and node.timestamp < cur - age
        ]
        if remove:
            self._summarize(remove)
            self._remove_nodes(remove)
        return remove

    def _remove_nodes(self, nodes: Sequence[int]) -> None:
        assert self.graph is not None
        to_remove = set(nodes)
        for n in to_remove:
            self.graph.nodes.pop(n, None)
            self.graph.edges.pop(n, None)
        for src, dsts in list(self.graph.edges.items()):
            self.graph.edges[src] = [d for d in dsts if d not in to_remove]
        self.graph.edge_timestamps = {
            (s, d): ts
            for (s, d), ts in self.graph.edge_timestamps.items()
            if s not in to_remove and d not in to_remove
        }
        _log_events(self.telemetry, "graph_prune", nodes, "node")

    def prune_if_needed(self, max_nodes: int) -> None:
        if self.graph is None:
            return
        if len(self.graph.nodes) > max_nodes:
            self.prune_low_degree()
            self.prune_old_nodes()


class MemoryPruningManager:
    """Monitor memory usage and prune rarely accessed vectors."""

    def __init__(
        self,
        threshold: int = 1,
        summarizer: Any | None = None,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self.threshold = int(threshold)
        self.summarizer = summarizer
        self.telemetry = telemetry
        self.memory: "HierarchicalMemory | None" = None

    def attach(self, memory: "HierarchicalMemory") -> None:
        """Attach to a :class:`HierarchicalMemory` instance."""
        self.memory = memory

    def _get_vector(self, meta: Any) -> np.ndarray | None:
        if self.memory is None:
            return None
        store = self.memory.store
        mlist = getattr(store, "_meta", getattr(store, "meta", []))
        try:
            idx = mlist.index(meta)
        except ValueError:
            return None
        vecs = getattr(store, "_vectors", getattr(store, "vectors", None))
        if vecs is None:
            return None
        if isinstance(vecs, list):
            if not vecs:
                return None
            mat = np.concatenate(vecs, axis=0)
        else:
            mat = vecs
        if idx >= len(mat):
            return None
        return mat[idx]

    def prune(self) -> None:
        """Prune or summarize entries below the usage ``threshold``."""
        if self.memory is None:
            return
        removed: List[Any] = []
        for meta, count in list(self.memory._usage.items()):
            if count >= self.threshold:
                continue
            vec = self._get_vector(meta)
            if self.summarizer is not None and vec is not None:
                comp = torch.from_numpy(vec).unsqueeze(0)
                full = self.memory.compressor.decoder(comp).squeeze(0)
                if hasattr(self.summarizer, "summarize"):
                    text = self.summarizer.summarize(full)
                else:
                    text = self.summarizer(full)
                self.memory.store.delete(tag=meta)
                zero = np.zeros((1, vec.shape[-1]), dtype=np.float32)
                self.memory.store.add(zero, [{"summary": text}])
            else:
                self.memory.store.delete(tag=meta)
            self.memory._usage.pop(meta, None)
            removed.append(meta)
        _log_events(self.telemetry, "memory_prune", removed, "meta")


__all__ = ["GraphPruningManager", "MemoryPruningManager"]
