"""Context summarization memory.

Store compressed summaries for distant tokens and restore them on retrieval.
"""
from __future__ import annotations

from typing import Iterable, Any, Tuple, List

import torch

from .hierarchical_memory import HierarchicalMemory


class ContextSummaryMemory(HierarchicalMemory):
    """Hierarchical memory that replaces far-past vectors with summaries."""

    def __init__(
        self,
        *args,
        summarizer,
        context_size: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
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
            self.store.delete(tag=meta)
            self.store.add(torch.zeros_like(vec).numpy(), [f"ctxsum:{summary}"])

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[Any]]:  # type: ignore[override]
        vecs, meta = super().search(query, k)
        new_vecs = []
        for v, m in zip(vecs, meta):
            if isinstance(m, str) and m.startswith("ctxsum:"):
                text = m.split(":", 1)[1]
                new_vecs.append(self.summarizer.expand(text).to(query.device))
            else:
                new_vecs.append(v)
        if new_vecs:
            vecs = torch.stack(new_vecs)
        return vecs, meta


__all__ = ["ContextSummaryMemory"]
