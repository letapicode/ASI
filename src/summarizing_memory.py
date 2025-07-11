"""Compress rarely used vectors with text summaries.

This module defines :class:`BaseSummarizingMemory` and its concrete
extension :class:`SummarizingMemory`.  The base class previously lived in
``summarizing_memory_base.py`` but has been merged here for simplicity.
"""

from __future__ import annotations

from typing import Iterable, Any, List

import torch

from .hierarchical_memory import HierarchicalMemory


class BaseSummarizingMemory(HierarchicalMemory):
    """Hierarchical memory that can replace seldom used vectors with summaries."""

    def __init__(
        self,
        *args: Any,
        summarizer: Any | None = None,
        summary_threshold: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.summarizer = summarizer
        self.summary_threshold = summary_threshold
        self.usage: List[int] = []

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        super().add(x, metadata)
        self.usage.extend([0] * len(x))

    def search(self, query: torch.Tensor, k: int = 5, **kwargs: Any):
        vecs, meta = super().search(query, k, **kwargs)
        for i in range(min(len(meta), len(self.usage))):
            self.usage[i] += 1
        return vecs, meta

    def summarize_rare(self) -> None:
        """Replace vectors with summaries when usage is below ``summary_threshold``."""
        if self.summarizer is None:
            return
        keep_vecs = []
        keep_meta = []
        keep_usage = []
        for vec, use, meta in zip(self.compressor.buffer.data, self.usage, self.store._meta):
            if use >= self.summary_threshold:
                keep_vecs.append(vec)
                keep_meta.append(meta)
                keep_usage.append(use)
            else:
                self._summarize_vector(vec, meta)
        self.compressor.buffer.data = keep_vecs
        self.store._meta = keep_meta
        self.usage = keep_usage

    # ------------------------------------------------------------------
    # internal helpers
    def _call_summarizer(self, vec: torch.Tensor) -> str:
        if hasattr(self.summarizer, "summarize"):
            return self.summarizer.summarize(vec)
        return self.summarizer(vec)

    def _summarize_vector(self, vec: torch.Tensor, meta: Any) -> None:
        text = self._call_summarizer(vec.unsqueeze(0))
        comp_dim = self.compressor.encoder.out_features
        new_vec = torch.zeros(comp_dim, dtype=vec.dtype)
        self.store.delete(tag=meta)
        self.store.add(new_vec.numpy(), [f"summary:{text}"])


class SummarizingMemory(BaseSummarizingMemory):
    """HierarchicalMemory with summarization of infrequent vectors."""

    def __init__(self, *args: Any, summary_threshold: int = 5, summarizer: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, summarizer=summarizer, summary_threshold=summary_threshold, **kwargs)

    def summarize(self, summarizer: Any | None = None) -> None:
        """Replace rarely used vectors with summaries."""
        if summarizer is not None:
            self.summarizer = summarizer
        self.summarize_rare()


__all__ = ["BaseSummarizingMemory", "SummarizingMemory"]
