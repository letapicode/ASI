import numpy as np
import torch
from typing import Iterable, Any, Tuple, List

from .hierarchical_memory import HierarchicalMemory
from .data_ingest import CrossLingualTranslator


def _embed_text(text: str, dim: int) -> torch.Tensor:
    """Deterministically embed text using a hash based RNG."""
    seed = abs(hash(text)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return torch.from_numpy(vec)


class CrossLingualMemory(HierarchicalMemory):
    """HierarchicalMemory wrapper storing translations for text input."""

    def __init__(
        self,
        *args: Any,
        translator: CrossLingualTranslator | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, translator=translator, **kwargs)
        self.text_dim = self.compressor.encoder.in_features

    def add_texts(
        self, texts: Iterable[str], metadata: Iterable[Any] | None = None
    ) -> None:
        """Embed ``texts`` and add them plus translations."""
        base = list(texts)
        metas = list(metadata) if metadata is not None else base
        vecs: List[torch.Tensor] = []
        out_meta: List[Any] = []
        for t, m in zip(base, metas):
            v = _embed_text(t, self.text_dim)
            vecs.append(v)
            out_meta.append(m)
            if self.translator is not None:
                for trans in self.translator.translate_all(t).values():
                    vecs.append(_embed_text(trans, self.text_dim))
                    out_meta.append(trans)
        stacked = torch.stack(vecs)
        super().add(stacked, out_meta)

    # ------------------------------------------------------------------
    # Convenience wrappers

    def add(
        self, x: torch.Tensor | str, metadata: Iterable[Any] | None = None
    ) -> None:  # type: ignore[override]
        """Add embeddings or raw text with translations."""
        if isinstance(x, str):
            self.add_texts([x], metadata)
        else:
            super().add(x, metadata)

    def search_text(self, text: str, k: int = 5) -> Tuple[torch.Tensor, List[Any]]:
        """Search for ``text`` across languages."""
        queries = [text]
        if self.translator is not None:
            queries += list(self.translator.translate_all(text).values())
        results: List[tuple[float, torch.Tensor, Any]] = []
        for q in queries:
            q_vec = _embed_text(q, self.text_dim)
            vecs, meta = super().search(q_vec, k)
            scores = torch.nn.functional.cosine_similarity(
                vecs, q_vec.expand_as(vecs), dim=1
            )
            results.extend([(s.item(), v, m) for s, v, m in zip(scores, vecs, meta)])
        if not results:
            return torch.empty(0, self.text_dim), []
        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:k]
        out_vecs = torch.stack([r[1] for r in top])
        out_meta = [r[2] for r in top]
        return out_vecs, out_meta

    def search(
        self, query: torch.Tensor | str, k: int = 5
    ) -> Tuple[torch.Tensor, List[Any]]:  # type: ignore[override]
        """Search by embedding or text."""
        if isinstance(query, str):
            return self.search_text(query, k)
        return super().search(query, k)


__all__ = ["CrossLingualMemory"]
