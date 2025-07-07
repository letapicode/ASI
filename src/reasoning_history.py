from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Sequence
import json
from collections import Counter

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .data_ingest import CrossLingualTranslator


@dataclass
class ReasoningHistoryLogger:
    """Store reasoning summaries with timestamps."""

    entries: List[Tuple[str, Any]] = field(default_factory=list)
    translator: CrossLingualTranslator | None = None

    def log(
        self,
        summary: str | Dict[str, Any],
        *,
        nodes: Sequence[int] | None = None,
        location: Any | None = None,
    ) -> None:
        ts = datetime.utcnow().isoformat()
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

    def get_history(self) -> List[Tuple[str, Any]]:
        return list(self.entries)

    def save(self, path: str) -> None:
        """Save history entries to ``path`` as JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh)

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
        """Cluster steps and detect contradictions."""
        counts: Counter[str] = Counter()
        contradictions: set[Tuple[str, str]] = set()
        for _ts, summary in self.entries:
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
        return HistoryAnalysis(dict(counts), sorted(contradictions))


@dataclass
class HistoryAnalysis:
    clusters: Dict[str, int]
    inconsistencies: List[Tuple[str, str]]


__all__ = ["ReasoningHistoryLogger", "HistoryAnalysis"]
