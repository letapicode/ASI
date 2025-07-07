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

    @staticmethod
    def summarize_multimodal(
        query: torch.Tensor,
        results: torch.Tensor,
        scores: List[float],
        provenance: List[Any],
    ) -> str:
        """Return a short description of multimodal provenance."""

        def _fmt(p: Any) -> str:
            if not isinstance(p, dict):
                return str(p)
            fields = []
            if (txt := p.get("text")):
                snippet = str(txt)
                if len(snippet) > 40:
                    snippet = snippet[:37] + "..."
                fields.append(f"text='{snippet}'")
            if (img := p.get("image")):
                fields.append(f"image={img}")
            if (aud := p.get("audio")):
                fields.append(f"audio={aud}")
            return ", ".join(fields) if fields else str(p)

        parts = [
            f"{i}. {_fmt(prov)} (score={score:.3f})"
            for i, (score, prov) in enumerate(zip(scores, provenance), start=1)
        ]
        return " | ".join(parts)

__all__ = ["RetrievalExplainer"]
