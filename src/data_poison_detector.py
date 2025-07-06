from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, List

import numpy as np
try:  # optional heavy dep
    from sklearn.cluster import KMeans
    _HAS_SK = True
except Exception:  # pragma: no cover - optional
    KMeans = None  # type: ignore
    _HAS_SK = False


class DataPoisonDetector:
    """Detect poisoned text samples using simple heuristics."""

    def __init__(self, window: int = 20, clusters: int = 4, threshold: float = 2.0) -> None:
        self.window = window
        self.clusters = clusters
        self.threshold = threshold
        self.vocab_sizes: Deque[int] = deque(maxlen=window)
        self.features: List[List[float]] = []
        self.model: KMeans | None = None

    # ------------------------------------------------------------------
    def _features(self, text: str) -> np.ndarray:
        words = text.split()
        vocab = set(words)
        avg_len = float(np.mean([len(w) for w in words]) if words else 0.0)
        digit_ratio = sum(c.isdigit() for c in text) / (len(text) + 1e-8)
        return np.array([len(vocab), avg_len, digit_ratio], dtype=float)

    # ------------------------------------------------------------------
    def record_text(self, text: str) -> bool:
        """Return ``True`` if ``text`` appears poisoned."""
        feat = self._features(text)
        vocab_size = int(feat[0])
        first = len(self.vocab_sizes) == 0
        avg_vocab = (
            sum(self.vocab_sizes) / len(self.vocab_sizes)
            if self.vocab_sizes
            else 1.0
        )
        self.vocab_sizes.append(vocab_size)
        self.features.append(list(feat))

        if _HAS_SK and self.model is None and len(self.features) >= self.clusters:
            self.model = KMeans(n_clusters=self.clusters, n_init="auto")
            self.model.fit(self.features)

        poisoned = False
        if vocab_size > avg_vocab * self.threshold:
            poisoned = True
        if first and vocab_size > 50:
            poisoned = True

        if self.model is not None:
            centers = self.model.cluster_centers_
            dist = np.min(np.linalg.norm(centers - feat, axis=1))
            dists = np.linalg.norm(self.model.transform(self.features), axis=1)
            avg_dist = float(dists.mean()) if dists.size else 0.0
            if avg_dist > 0 and dist > avg_dist * self.threshold:
                poisoned = True

        return poisoned

    # ------------------------------------------------------------------
    def record_file(self, path: str | Path) -> bool:
        try:
            text = Path(path).read_text()
        except Exception:
            return False
        return self.record_text(text)


__all__ = ["DataPoisonDetector"]
