from __future__ import annotations

import numpy as np
from typing import Sequence


def sample_actions_qae(logits: Sequence[float]) -> int:
    """Mock quantum amplitude estimation sampler."""
    arr = np.asarray(logits, dtype=float)
    probs = np.exp(arr) / np.exp(arr).sum()
    return int(np.random.choice(len(probs), p=probs))


__all__ = ["sample_actions_qae"]
