"""Generate weak labels for unlabeled triples."""

from __future__ import annotations

from typing import Iterable


class AutoLabeler:
    """Simple heuristic labeler using word counts."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def label(self, texts: Iterable[str]) -> list[int]:
        labels = []
        for t in texts:
            val = sum(ord(c) for c in t) % self.vocab_size
            labels.append(val)
        return labels


__all__ = ["AutoLabeler"]
