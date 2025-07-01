"""Dataset filtering using generative noise detection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List
import math


class AutoDatasetFilter:
    """Simple unigram language model for noise detection."""

    def __init__(self, threshold: float = -3.0) -> None:
        self.threshold = threshold
        self.probs: dict[str, float] | None = None

    def fit(self, texts: Iterable[str]) -> None:
        """Fit a unigram model on the provided ``texts``."""
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(t)
        total = sum(counter.values())
        if total == 0:
            self.probs = None
            return
        self.probs = {c: n / total for c, n in counter.items()}

    def score(self, text: str) -> float:
        """Return average log probability of ``text`` under the model."""
        if not self.probs:
            raise ValueError("model not trained")
        ll = 0.0
        for ch in text:
            ll += math.log(self.probs.get(ch, 1e-6))
        if len(text) == 0:
            return float("-inf")
        return ll / len(text)

    def prune(self, texts: Iterable[str]) -> List[str]:
        """Return subset of ``texts`` scoring above the threshold."""
        text_list = list(texts)
        self.fit(text_list)
        if not self.probs:
            return text_list
        return [t for t in text_list if self.score(t) >= self.threshold]


def filter_text_files(text_paths: Iterable[str | Path], threshold: float = -3.0) -> List[Path]:
    """Filter text files in ``text_paths`` using :class:`AutoDatasetFilter`."""
    paths = [Path(p) for p in text_paths]
    texts = [p.read_text() for p in paths]
    filt = AutoDatasetFilter(threshold=threshold)
    filt.fit(texts)
    if not filt.probs:
        return paths
    return [p for p, t in zip(paths, texts) if filt.score(t) >= threshold]


__all__ = ["AutoDatasetFilter", "filter_text_files"]
