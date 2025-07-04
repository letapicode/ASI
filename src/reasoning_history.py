from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple


@dataclass
class ReasoningHistoryLogger:
    """Store reasoning summaries with timestamps."""

    entries: List[Tuple[str, str]] = field(default_factory=list)

    def log(self, summary: str) -> None:
        ts = datetime.utcnow().isoformat()
        self.entries.append((ts, summary))

    def get_history(self) -> List[Tuple[str, str]]:
        return list(self.entries)


__all__ = ["ReasoningHistoryLogger"]
