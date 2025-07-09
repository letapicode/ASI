"""Compress rarely used vectors with text summaries."""

from __future__ import annotations

from typing import Iterable, Any

import torch

from .summarizing_memory_base import BaseSummarizingMemory


class SummarizingMemory(BaseSummarizingMemory):
    """HierarchicalMemory with summarization of infrequent vectors."""

    def __init__(self, *args: Any, summary_threshold: int = 5, summarizer: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, summarizer=summarizer, summary_threshold=summary_threshold, **kwargs)

    def summarize(self, summarizer: Any | None = None) -> None:
        """Replace rarely used vectors with summaries."""
        if summarizer is not None:
            self.summarizer = summarizer
        self.summarize_rare()


__all__ = ["SummarizingMemory"]
