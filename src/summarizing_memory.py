"""Compress rarely used vectors with text summaries."""

from __future__ import annotations

from typing import Iterable, Any, List

import torch

from .hierarchical_memory import HierarchicalMemory


class SummarizingMemory(HierarchicalMemory):
    """HierarchicalMemory with summarization of infrequent vectors."""

    def __init__(self, *args, summary_threshold: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.summary_threshold = summary_threshold
        self.usage: List[int] = []

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:  # type: ignore[override]
        super().add(x, metadata)
        self.usage.extend([0] * len(x))

    def search(self, query: torch.Tensor, k: int = 5):  # type: ignore[override]
        vecs, meta = super().search(query, k)
        for i, m in enumerate(meta):
            if i < len(self.usage):
                self.usage[i] += 1
        return vecs, meta

    def summarize(self, summarizer) -> None:
        """Replace rarely used vectors with summaries from ``summarizer``."""
        keep = []
        keep_meta = []
        new = []
        for vec, use, meta in zip(self.compressor.buffer.data, self.usage, self.store._meta):
            if use >= self.summary_threshold:
                keep.append(vec)
                keep_meta.append(meta)
            else:
                text = summarizer(vec.unsqueeze(0))
                new_vec = torch.zeros_like(vec)
                self.store.delete(tag=meta)
                self.store.add(new_vec.numpy(), [f"summary:{text}"])
        self.compressor.buffer.data = keep
        self.store._meta = keep_meta
        self.usage = [u for u in self.usage if u >= self.summary_threshold]


__all__ = ["SummarizingMemory"]
