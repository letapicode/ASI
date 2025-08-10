"""Compute and mitigate dataset bias.

This module combines helpers for detecting representation skews and
filtering datasets using those metrics.  Previously the detection and
mitigation logic lived in separate files which led to duplicated imports
and scattered responsibilities.  Consolidating them here keeps all
bias-related utilities in one place and avoids cross-file redundancy."""

from __future__ import annotations

import numpy as np
from collections import Counter
import concurrent.futures
from pathlib import Path
from typing import Iterable, Dict, Union, Callable, List, Tuple, Optional

from .telemetry import TelemetryLogger


def compute_word_freq(
    text_files: Iterable[str | Path], num_workers: int | None = None
) -> Dict[str, int]:
    """Return word frequencies aggregated across ``text_files``."""

    def _load(p: str | Path) -> Counter[str]:
        return Counter(Path(p).read_text().split())

    counter: Counter[str] = Counter()
    if num_workers and num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            for c in ex.map(_load, text_files):
                counter.update(c)
    else:
        for p in text_files:
            counter.update(_load(p))
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


class DatasetBiasDetector:
    """Compute and stream bias metrics during ingestion."""

    def __init__(self, telemetry: TelemetryLogger | None = None) -> None:
        self.telemetry = telemetry or TelemetryLogger(interval=0.5)
        self.scores: List[float] = []
        self.callbacks: List[Callable[[float], None]] = []
        self.cache: Dict[str, float] = {}

    # --------------------------------------------------------------
    def add_callback(self, cb: Callable[[float], None]) -> None:
        self.callbacks.append(cb)

    # --------------------------------------------------------------
    def score_file(self, path: str | Path) -> float:
        key = str(path)
        if key in self.cache:
            score = self.cache[key]
        else:
            score = file_bias_score(path)
            self.cache[key] = score
        self.scores.append(score)
        self._update(score)
        return score

    # --------------------------------------------------------------
    def _update(self, score: float) -> None:
        avg = float(sum(self.scores) / len(self.scores))
        self.telemetry.metrics["bias_score"] = score
        self.telemetry.metrics["bias_avg"] = avg
        for cb in list(self.callbacks):
            try:
                cb(score)
            except Exception:
                pass

    # --------------------------------------------------------------
    def stream_metrics(self) -> Iterable[Dict[str, float]]:
        for i, s in enumerate(self.scores, start=1):
            yield {
                "bias_score": s,
                "bias_avg": float(sum(self.scores[:i]) / i),
            }


class DataBiasMitigator:
    """Reweight or filter dataset samples based on bias scores."""

    def __init__(self, threshold: float | None = None) -> None:
        """Initialize with optional ``threshold`` for filtering."""
        self.threshold = threshold

    # ------------------------------------------------------------------
    def score_file(self, path: str | Path) -> float:
        """Return bias score for the file at ``path``."""
        return file_bias_score(path)

    # ------------------------------------------------------------------
    def reweight_files(self, paths: Iterable[str | Path]) -> Dict[Path, float]:
        """Return a weight for each file proportional to its bias score."""
        return {Path(p): self.score_file(p) for p in paths}

    # ------------------------------------------------------------------
    def filter_files(
        self,
        paths: Iterable[str | Path],
        threshold: Optional[float] = None,
    ) -> List[Path]:
        """Return paths whose bias score meets ``threshold``."""
        thr = self.threshold if threshold is None else threshold
        if thr is None:
            return [Path(p) for p in paths]
        return [Path(p) for p in paths if self.score_file(p) >= thr]

    # ------------------------------------------------------------------
    def apply_to_triples(
        self,
        triples: Iterable[Tuple[Path, Path, Path]],
        threshold: Optional[float] = None,
    ) -> List[Tuple[Path, Path, Path]]:
        """Filter triples based on text bias score."""
        thr = self.threshold if threshold is None else threshold
        out: List[Tuple[Path, Path, Path]] = []
        for t, i, a in triples:
            score = self.score_file(t)
            if thr is None or score >= thr:
                out.append((t, i, a))
        return out


__all__ = [
    "compute_word_freq",
    "bias_score",
    "text_bias_score",
    "file_bias_score",
    "DatasetBiasDetector",
    "DataBiasMitigator",
]
