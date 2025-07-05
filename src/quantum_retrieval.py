from __future__ import annotations

from typing import Sequence
import numpy as np


def amplify_search(scores: Sequence[float], k: int = 1) -> list[int]:
    """Mock amplitude amplification over similarity scores.

    Parameters
    ----------
    scores : Sequence[float]
        Similarity scores to convert into retrieval probabilities.
    k : int, optional
        Number of indices to sample, by default ``1``.

    Returns
    -------
    list[int]
        Selected indices ordered by decreasing amplified probability.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return []
    probs = np.exp(arr)
    probs /= probs.sum()
    amp_probs = probs ** 2
    amp_probs /= amp_probs.sum()
    k = min(k, len(arr))
    idx = np.random.choice(len(arr), size=k, replace=False, p=amp_probs)
    order = np.argsort(amp_probs[idx])[::-1]
    return [int(idx[i]) for i in order]


__all__ = ["amplify_search"]
