from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional, Union
import json


@dataclass
class TimedTriple:
    """Triple with optional ``timestamp``."""

    subject: str
    predicate: str
    object: str
    timestamp: Optional[float] = None


class KnowledgeGraphMemory:
    """Store triples of the form ``(subject, predicate, object)``."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.triples: List[Tuple[str, str, str]] = []
        self.timestamps: List[Optional[float]] = []
        self.path = Path(path) if path is not None else None
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            for item in data:
                if len(item) == 4:
                    s, p, o, ts = item
                    self.triples.append((s, p, o))
                    self.timestamps.append(ts)
                else:
                    self.triples.append(tuple(item))
                    self.timestamps.append(None)

    # ------------------------------------------------------------
    def add_triples(
        self, triples: Iterable[Union[Tuple[str, str, str], Tuple[str, str, str, float], TimedTriple]]
    ) -> None:
        """Add triples to the store."""
        for t in triples:
            ts: Optional[float] = None
            if isinstance(t, TimedTriple):
                s, p, o, ts = t.subject, t.predicate, t.object, t.timestamp
            else:
                if len(t) not in {3, 4}:
                    raise ValueError("triples must be (subject, predicate, object) or with timestamp")
                s, p, o = map(str, t[:3])
                if len(t) == 4:
                    ts = float(t[3])
            self.triples.append((s, p, o))
            self.timestamps.append(ts)
        if self.path:
            data = [list(t) + ([ts] if ts is not None else []) for t, ts in zip(self.triples, self.timestamps)]
            self.path.write_text(json.dumps(data))

    # ------------------------------------------------------------
    def query_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[TimedTriple]:
        """Return triples matching the provided pattern and time range."""
        out: List[TimedTriple] = []
        for (s, p, o), ts in zip(self.triples, self.timestamps):
            if subject is not None and s != subject:
                continue
            if predicate is not None and p != predicate:
                continue
            if object is not None and o != object:
                continue
            if start_time is not None or end_time is not None:
                if ts is None:
                    continue
                if start_time is not None and ts < start_time:
                    continue
                if end_time is not None and ts > end_time:
                    continue
            out.append(TimedTriple(s, p, o, ts))
        return out


__all__ = ["KnowledgeGraphMemory", "TimedTriple"]
