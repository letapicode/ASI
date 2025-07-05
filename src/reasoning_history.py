from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple
import json
from collections import Counter


@dataclass
class ReasoningHistoryLogger:
    """Store reasoning summaries with timestamps."""

    entries: List[Tuple[str, str]] = field(default_factory=list)

    def log(self, summary: str) -> None:
        ts = datetime.utcnow().isoformat()
        self.entries.append((ts, summary))

    def get_history(self) -> List[Tuple[str, str]]:
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
            logger.entries.append((ts, summary))
        return logger

    def analyze(self) -> "HistoryAnalysis":
        """Cluster steps and detect contradictions."""
        counts: Counter[str] = Counter()
        contradictions: set[Tuple[str, str]] = set()
        for _ts, summary in self.entries:
            steps = [s.strip() for s in summary.split("->")]
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
