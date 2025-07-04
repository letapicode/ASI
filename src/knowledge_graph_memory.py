from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional
import json


class KnowledgeGraphMemory:
    """Store triples of the form ``(subject, predicate, object)``."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.triples: List[Tuple[str, str, str]] = []
        self.path = Path(path) if path is not None else None
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            self.triples = [tuple(t) for t in data]

    # ------------------------------------------------------------
    def add_triples(self, triples: Iterable[Tuple[str, str, str]]) -> None:
        """Add triples to the store."""
        for t in triples:
            if len(t) != 3:
                raise ValueError("triples must be (subject, predicate, object)")
            self.triples.append(tuple(map(str, t)))
        if self.path:
            self.path.write_text(json.dumps(self.triples))

    # ------------------------------------------------------------
    def query_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[Tuple[str, str, str]]:
        """Return triples matching the provided pattern."""
        out: List[Tuple[str, str, str]] = []
        for s, p, o in self.triples:
            if subject is not None and s != subject:
                continue
            if predicate is not None and p != predicate:
                continue
            if object is not None and o != object:
                continue
            out.append((s, p, o))
        return out


__all__ = ["KnowledgeGraphMemory"]
