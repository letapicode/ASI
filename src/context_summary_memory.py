"""Context summarization memory.

Store compressed summaries for distant tokens and restore them on retrieval.
"""
from __future__ import annotations

from typing import Iterable, Any, Tuple, List, Dict

import torch

from .hierarchical_memory import HierarchicalMemory
from .data_ingest import CrossLingualTranslator


class ContextSummaryMemory(HierarchicalMemory):
    """Hierarchical memory that replaces far-past vectors with summaries."""

    def __init__(
        self,
        *args,
        summarizer,
        context_size: int = 1024,
        translator: CrossLingualTranslator | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, translator=translator, **kwargs)
        self.summarizer = summarizer
        self.context_size = context_size

    def summarize_context(self) -> None:
        """Compress vectors beyond ``context_size`` using ``summarizer``."""
        total = len(self.compressor.buffer.data)
        if total <= self.context_size:
            return
        keep_start = total - self.context_size
        old_vecs = self.compressor.buffer.data[:keep_start]
        old_meta = self.store._meta[:keep_start]
        self.compressor.buffer.data = self.compressor.buffer.data[keep_start:]
        self.store._meta = self.store._meta[keep_start:]
        for vec, meta in zip(old_vecs, old_meta):
            summary = self.summarizer.summarize(vec.unsqueeze(0))
            info: Dict[str, Any] = {"summary": summary}
            if self.translator is not None:
                info["translations"] = self.translator.translate_all(summary)
            self.store.delete(tag=meta)
            self.store.add(torch.zeros_like(vec).numpy(), [{"ctxsum": info}])

    def search(
        self, query: torch.Tensor, k: int = 5, language: str | None = None
    ) -> Tuple[torch.Tensor, List[Any]]:  # type: ignore[override]
        vecs, meta = super().search(query, k)
        new_vecs = []
        out_meta: List[Any] = []
        for v, m in zip(vecs, meta):
            if isinstance(m, dict) and "ctxsum" in m:
                info = m["ctxsum"]
                text = info["summary"]
                new_vecs.append(self.summarizer.expand(text).to(query.device))
                if language is not None and self.translator is not None:
                    trans = info.get("translations", {}).get(language)
                    if trans is None:
                        trans = self.translator.translate(text, language)
                    out_meta.append(trans)
                else:
                    out_meta.append(m)
            elif isinstance(m, str) and m.startswith("ctxsum:"):
                text = m.split(":", 1)[1]
                new_vecs.append(self.summarizer.expand(text).to(query.device))
                out_meta.append(m)
            else:
                new_vecs.append(v)
                out_meta.append(m)
        if new_vecs:
            vecs = torch.stack(new_vecs)
        return vecs, out_meta


__all__ = ["ContextSummaryMemory"]
