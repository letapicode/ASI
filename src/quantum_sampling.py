from __future__ import annotations

from typing import Sequence
import numpy as np


def _softmax(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    arr = arr - arr.max()
    probs = np.exp(arr)
    probs /= probs.sum()
    return probs


def amplify_search(
    scores: Sequence[float],
    k: int = 1,
    lang_tags: Sequence[str] | None = None,
) -> list[int]:
    """Mock amplitude amplification over similarity scores."""
    probs = _softmax(scores)
    if probs.size == 0:
        return []
    if lang_tags is not None:
        if len(lang_tags) != len(probs):
            raise ValueError("lang_tags length mismatch")
        counts: dict[str, int] = {}
        for t in lang_tags:
            counts[t] = counts.get(t, 0) + 1
        weights = np.array([1.0 / counts[t] for t in lang_tags], dtype=float)
        probs *= weights
        probs /= probs.sum()
    amp_probs = probs ** 2
    amp_probs /= amp_probs.sum()
    k = min(k, len(amp_probs))
    idx = np.random.choice(len(amp_probs), size=k, replace=False, p=amp_probs)
    order = np.argsort(amp_probs[idx])[::-1]
    return [int(idx[i]) for i in order]


def sample_actions_qae(logits: Sequence[float]) -> int:
    """Mock quantum amplitude estimation sampler."""
    probs = _softmax(logits)
    return int(np.random.choice(len(probs), p=probs))


__all__ = ["amplify_search", "sample_actions_qae"]
