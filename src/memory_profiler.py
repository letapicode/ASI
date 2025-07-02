"""Simple profiling hooks for HierarchicalMemory."""

from __future__ import annotations

import json
import time
from io import StringIO
from typing import Any, Callable, Dict, Tuple

import torch

from .hierarchical_memory import HierarchicalMemory


class MemoryProfiler:
    """Record query statistics from a HierarchicalMemory instance."""

    def __init__(self, memory: HierarchicalMemory) -> None:
        self.memory = memory
        self.reset()
        self._orig_search: Callable[..., Tuple[torch.Tensor, list]] | None = None
        self._orig_asearch: Callable[..., Any] | None = None

    def reset(self) -> None:
        """Reset all counters."""
        self.query_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.total_latency = 0.0

    # --------------------------------------------------------------
    def start_profiling(self) -> None:
        """Patch the memory instance so searches update stats."""
        if self._orig_search is not None:
            return

        self._orig_search = self.memory.search

        def search_wrapper(query: torch.Tensor, k: int = 5):
            start = time.perf_counter()
            out, meta = self._orig_search(query, k)
            elapsed = time.perf_counter() - start
            self.query_count += 1
            self.total_latency += elapsed
            if meta:
                self.hit_count += 1
            else:
                self.miss_count += 1
            return out, meta

        self.memory.search = search_wrapper  # type: ignore[assignment]

        if hasattr(self.memory, "asearch"):
            self._orig_asearch = self.memory.asearch

            async def asearch_wrapper(query: torch.Tensor, k: int = 5):
                start = time.perf_counter()
                out, meta = await self._orig_asearch(query, k)
                elapsed = time.perf_counter() - start
                self.query_count += 1
                self.total_latency += elapsed
                if meta:
                    self.hit_count += 1
                else:
                    self.miss_count += 1
                return out, meta

            self.memory.asearch = asearch_wrapper  # type: ignore[assignment]

    # --------------------------------------------------------------
    def report_stats(self, fmt: str = "json") -> str:
        """Return collected metrics in ``json`` or ``csv`` format."""
        avg_latency = self.total_latency / self.query_count if self.query_count else 0.0
        stats: Dict[str, Any] = {
            "queries": self.query_count,
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": self.hit_count / self.query_count if self.query_count else 0.0,
            "avg_latency": avg_latency,
        }
        if fmt.lower() == "csv":
            buf = StringIO()
            buf.write(",".join(["queries", "hits", "misses", "hit_rate", "avg_latency"]))
            buf.write("\n")
            buf.write(
                f"{stats['queries']},{stats['hits']},{stats['misses']},{stats['hit_rate']},{stats['avg_latency']}"
            )
            return buf.getvalue()
        return json.dumps(stats)


__all__ = ["MemoryProfiler"]
