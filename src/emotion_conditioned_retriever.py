from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any, Iterable, Tuple, List

from .emotion_detector import detect_emotion

try:  # optional torch dependency
    import torch
except Exception:  # pragma: no cover - allow running without torch
    torch = None

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .hierarchical_memory import HierarchicalMemory


def _embed_text(text: str, dim: int) -> Any:
    """Deterministically embed ``text`` using a hash RNG."""
    seed = abs(hash(text)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    if torch is not None:
        return torch.from_numpy(vec)
    return vec


def _cosine_similarity(vecs: Any, query: Any) -> List[float]:
    if torch is not None and isinstance(vecs, torch.Tensor):
        q = query.expand_as(vecs)
        sims = torch.nn.functional.cosine_similarity(vecs, q, dim=1)
        return sims.cpu().tolist()
    arr = np.asarray(vecs)
    q = np.asarray(query)
    denom = np.linalg.norm(arr, axis=1) * np.linalg.norm(q)
    denom[denom == 0] = 1.0
    sims = (arr @ q) / denom
    return sims.tolist()


class EmotionConditionedRetriever:
    """Rank memory search results by emotional relevance."""

    def __init__(self, memory: "HierarchicalMemory" | Any, *, dim: int | None = None) -> None:
        self.memory = memory
        if dim is not None:
            self.dim = dim
        else:
            self.dim = getattr(memory, "dim", None)
            if self.dim is None and hasattr(memory, "compressor"):
                self.dim = getattr(memory.compressor.encoder, "in_features", 0)
            if self.dim is None:
                self.dim = 0

    # ------------------------------------------------------------
    def search_with_emotion(self, query_text: str, k: int = 5) -> Tuple[Any, List[Any]]:
        """Return vectors ordered by matching the query emotion."""
        q_vec = _embed_text(query_text, self.dim)
        try:
            vecs, meta, scores = self.memory.search(q_vec, k=k, return_scores=True)
        except TypeError:
            vecs, meta = self.memory.search(q_vec, k=k)
            scores = _cosine_similarity(vecs, q_vec)
        query_emotion = detect_emotion(query_text)
        weights = []
        for m in meta:
            if isinstance(m, str):
                emo = detect_emotion(m)
                weights.append(1.0 if emo == query_emotion else 0.0)
            else:
                weights.append(0.0)
        final = [s + w for s, w in zip(scores, weights)]
        order = list(np.argsort(final)[::-1])
        if torch is not None and isinstance(vecs, torch.Tensor):
            vecs = vecs[order]
        else:
            vecs = np.asarray(vecs)[order]
        meta = [meta[i] for i in order]
        return vecs, meta


__all__ = ["EmotionConditionedRetriever", "_embed_text"]
