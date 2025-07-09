from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Sequence
import json
from collections import Counter
import sys
import types

if __name__ not in sys.modules:  # pragma: no cover - for manual loaders
    sys.modules[__name__] = types.ModuleType(__name__)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .data_ingest import CrossLingualTranslator
    from .graph_of_thought import GraphOfThought
    from .graph_pruning_manager import GraphPruningManager


@dataclass
class ReasoningHistoryLogger:
    """Store reasoning summaries with timestamps."""

    entries: List[Tuple[str, Any]] = field(default_factory=list)
    translator: CrossLingualTranslator | None = None
    pruner: "GraphPruningManager | None" = None
    prune_threshold: int = 0

    def log(
        self,
        summary: str | Dict[str, Any],
        *,
        nodes: Sequence[int] | None = None,
        location: Any | None = None,
    ) -> None:
        ts = datetime.now(UTC).isoformat()
        if isinstance(summary, dict):
            entry = dict(summary)
            if nodes is not None:
                entry["nodes"] = list(nodes)
            if location is not None:
                entry["location"] = location
            if self.translator is not None and "translations" not in entry and "summary" in entry:
                entry["translations"] = self.translator.translate_all(entry["summary"])
            if "image_vec" in entry and hasattr(entry["image_vec"], "tolist"):
                entry["image_vec"] = list(entry["image_vec"])
            if "audio_vec" in entry and hasattr(entry["audio_vec"], "tolist"):
                entry["audio_vec"] = list(entry["audio_vec"])
            self.entries.append((ts, entry))
        else:
            if self.translator is not None:
                entry = {
                    "summary": summary,
                    "translations": self.translator.translate_all(summary),
                }
                if nodes is not None:
                    entry["nodes"] = list(nodes)
                if location is not None:
                    entry["location"] = location
                self.entries.append((ts, entry))
            else:
                self.entries.append((ts, summary))

        if (
            self.pruner is not None
            and self.pruner.graph is not None
            and self.prune_threshold
            and len(self.pruner.graph.nodes) > self.prune_threshold
        ):
            self.pruner.prune_low_degree()
            self.pruner.prune_old_nodes()

    def get_history(self) -> List[Tuple[str, Any]]:
        return list(self.entries)

    def save(self, path: str) -> None:
        """Save history entries to ``path`` as JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh)

    def save_graph(self, path: str, graph: "GraphOfThought") -> None:
        """Persist ``graph`` to ``path`` and record the snapshot."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(graph.to_json(), fh)
        self.log({"graph_path": path})

    def log_debate(
        self, transcript: Sequence[Tuple[str, str]], verdict: str
    ) -> None:
        """Record a Socratic debate transcript and verdict."""
        ts = datetime.now(UTC).isoformat()
        entry = {"transcript": list(transcript), "verdict": verdict}
        self.entries.append((ts, entry))

    @classmethod
    def load(cls, path: str) -> "ReasoningHistoryLogger":
        """Load entries from ``path`` and return a logger."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger = cls()
        for ts, summary in data:
            if isinstance(summary, dict):
                if "image_vec" in summary:
                    summary["image_vec"] = list(summary["image_vec"])
                if "audio_vec" in summary:
                    summary["audio_vec"] = list(summary["audio_vec"])
            logger.entries.append((ts, summary))
        return logger

    def analyze(self) -> "HistoryAnalysis":
        """Cluster steps, detect contradictions and graph changes."""
        counts: Counter[str] = Counter()
        contradictions: set[Tuple[str, str]] = set()
        graph_paths: List[Tuple[str, str]] = []
        for ts, summary in self.entries:
            text = summary["summary"] if isinstance(summary, dict) else summary
            steps = [s.strip() for s in text.split("->")]
            for step in steps:
                counts[step] += 1
                neg = f"not {step}"
                if counts[neg]:
                    contradictions.add((step, neg))
                if step.startswith("not "):
                    base = step[4:]
                    if counts[base]:
                        contradictions.add((base, step))
            if isinstance(summary, dict) and "graph_path" in summary:
                graph_paths.append((ts, summary["graph_path"]))

        diffs: List[Dict[str, object]] = []
        prev: dict | None = None
        for ts, path in graph_paths:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if prev is not None:
                diff = _diff_graph_data(prev, data)
                if (
                    diff["added_nodes"]
                    or diff["changed_nodes"]
                    or diff["added_edges"]
                    or diff["changed_edges"]
                ):
                    diffs.append(
                        {
                            "timestamp": ts,
                            "added_nodes": len(diff["added_nodes"]),
                            "changed_nodes": len(diff["changed_nodes"]),
                            "added_edges": len(diff["added_edges"]),
                            "changed_edges": len(diff["changed_edges"]),
                        }
                    )
            prev = data

        return HistoryAnalysis(dict(counts), sorted(contradictions), diffs)


@dataclass
class HistoryAnalysis:
    clusters: Dict[str, int]
    inconsistencies: List[Tuple[str, str]]
    diffs: List[Dict[str, object]] = field(default_factory=list)


__all__ = ["ReasoningHistoryLogger", "HistoryAnalysis"]


def _diff_graph_data(old: dict, new: dict) -> Dict[str, object]:
    """Return added/changed nodes and edges between two graphs."""
    def _node_map(data: dict) -> Dict[str, dict]:
        return {
            n.get("stable_id", str(n.get("id"))): n for n in data.get("nodes", [])
        }

    def _id_map(data: dict) -> Dict[int, str]:
        return {
            int(n.get("id")): n.get("stable_id", str(n.get("id")))
            for n in data.get("nodes", [])
        }

    old_nodes = _node_map(old)
    new_nodes = _node_map(new)
    added_nodes = [n for sid, n in new_nodes.items() if sid not in old_nodes]
    changed_nodes = [
        sid
        for sid, n in new_nodes.items()
        if sid in old_nodes
        and (
            n.get("text") != old_nodes[sid].get("text")
            or n.get("metadata") != old_nodes[sid].get("metadata")
        )
    ]

    old_ids = _id_map(old)
    new_ids = _id_map(new)
    old_edges = {
        (old_ids.get(e[0], str(e[0])), old_ids.get(e[1], str(e[1]))): e
        for e in old.get("edges", [])
    }
    new_edges = {
        (new_ids.get(e[0], str(e[0])), new_ids.get(e[1], str(e[1]))): e
        for e in new.get("edges", [])
    }
    added_edges = [k for k in new_edges if k not in old_edges]
    changed_edges = [
        k for k in new_edges if k in old_edges and new_edges[k] != old_edges[k]
    ]
    return {
        "added_nodes": added_nodes,
        "changed_nodes": changed_nodes,
        "added_edges": added_edges,
        "changed_edges": changed_edges,
    }
