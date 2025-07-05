from __future__ import annotations
from typing import Any, List
import torch


class RetrievalExplainer:
    """Format retrieval traces for analysis."""

    @staticmethod
    def format(query: torch.Tensor, results: torch.Tensor, scores: List[float], provenance: List[Any]) -> List[dict]:
        items = []
        for s, p, r in zip(scores, provenance, results):
            items.append({"provenance": p, "score": float(s), "vector": r.tolist()})
        return items

    @staticmethod
    def summarize(
        query: torch.Tensor,
        results: torch.Tensor,
        scores: List[float],
        provenance: List[Any],
    ) -> str:
        """Return a short textual summary of retrieval output."""
        parts = []
        for i, (score, src) in enumerate(zip(scores, provenance), start=1):
            parts.append(f"{i}. {src} (score={score:.3f})")
        return " | ".join(parts)

__all__ = ["RetrievalExplainer"]
