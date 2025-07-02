from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List


class CausalGraphLearner:
    """Infer a simple directed graph from state transitions."""

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold
        self.adj: np.ndarray | None = None

    def fit(self, transitions: Iterable[Tuple[np.ndarray, int, np.ndarray]]) -> None:
        """Fit edges from ``(state, action, next_state)`` tuples."""
        states = []
        deltas = []
        for s, _a, ns in transitions:
            states.append(np.asarray(s, dtype=float))
            deltas.append(np.asarray(ns, dtype=float) - np.asarray(s, dtype=float))
        if not states:
            self.adj = None
            return
        S = np.stack(states)
        D = np.stack(deltas)
        data = np.concatenate([S, D], axis=1)
        corr = np.corrcoef(data, rowvar=False)
        n = S.shape[1]
        self.adj = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                coeff = corr[i, n + j]
                if abs(coeff) >= self.threshold:
                    self.adj[i, j] = coeff

    def edges(self) -> List[Tuple[int, int, float]]:
        """Return edges as ``(src, dst, weight)`` tuples."""
        if self.adj is None:
            return []
        out: List[Tuple[int, int, float]] = []
        n = self.adj.shape[0]
        for i in range(n):
            for j in range(n):
                w = float(self.adj[i, j])
                if w != 0.0:
                    out.append((i, j, w))
        return out


__all__ = ["CausalGraphLearner"]
