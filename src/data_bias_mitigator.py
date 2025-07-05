from __future__ import annotations

"""Mitigate dataset bias by reweighting or filtering text files."""

from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional

from .dataset_bias_detector import text_bias_score, file_bias_score


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


__all__ = ["DataBiasMitigator"]
