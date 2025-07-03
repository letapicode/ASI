"""Compute simple representation metrics for dataset bias analysis."""

from __future__ import annotations

import numpy as np
from collections import Counter
from pathlib import Path
from typing import Iterable, Dict, Union


def compute_word_freq(text_files: Iterable[str | Path]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for p in text_files:
        words = Path(p).read_text().split()
        counter.update(words)
    return dict(counter)


def bias_score(freq: Dict[str, int]) -> float:
    counts = np.array(list(freq.values()), dtype=float)
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    entropy = -(probs * np.log(probs + 1e-8)).sum()
    return entropy / np.log(len(counts) + 1e-8)


def text_bias_score(text: str) -> float:
    """Return the bias score for a single ``text``."""
    freq = Counter(text.split())
    return bias_score(freq)


def file_bias_score(path: Union[str, Path]) -> float:
    """Return the bias score for the text file at ``path``."""
    return text_bias_score(Path(path).read_text())


__all__ = [
    "compute_word_freq",
    "bias_score",
    "text_bias_score",
    "file_bias_score",
]
