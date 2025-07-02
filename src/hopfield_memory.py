from __future__ import annotations

import numpy as np


class HopfieldMemory:
    """Simple Hopfield network storing binary patterns."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.weights = np.zeros((dim, dim), dtype=np.float32)
        self._count = 0

    def store(self, patterns: np.ndarray) -> None:
        """Store one or more patterns represented with +/-1 values."""
        arr = np.asarray(patterns, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = np.where(arr >= 0, 1.0, -1.0)
        for p in arr:
            self.weights += np.outer(p, p)
            self._count += 1
        np.fill_diagonal(self.weights, 0.0)
        self.weights /= float(self.dim)

    def retrieve(self, query: np.ndarray, steps: int = 5) -> np.ndarray:
        """Recover a stored pattern from a noisy ``query``."""
        x = np.asarray(query, dtype=np.float32)
        squeeze = False
        if x.ndim == 1:
            x = x[None, :]
            squeeze = True
        x = np.where(x >= 0, 1.0, -1.0)
        for _ in range(max(1, steps)):
            new_x = np.sign(x @ self.weights)
            new_x[new_x == 0] = 1.0
            if np.allclose(new_x, x):
                x = new_x
                break
            x = new_x
        return x[0] if squeeze else x
