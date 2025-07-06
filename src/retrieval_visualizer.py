from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt


class RetrievalVisualizer:
    """Record retrieval hit/miss events and render simple plots."""

    def __init__(self, memory: Any) -> None:
        self.memory = memory
        self.log: List[Dict[str, float]] = []
        self._orig_search = None
        self._orig_asearch = None

    # --------------------------------------------------------------
    def start(self) -> None:
        if self._orig_search is not None:
            return
        self._orig_search = self.memory.search

        def search_wrapper(query, k: int = 5):
            t0 = time.perf_counter()
            out, meta = self._orig_search(query, k)
            latency = time.perf_counter() - t0
            self.log.append({
                "time": time.time(),
                "hit": float(bool(meta)),
                "latency": latency,
            })
            return out, meta

        self.memory.search = search_wrapper

        if hasattr(self.memory, "asearch"):
            self._orig_asearch = self.memory.asearch

            async def asearch_wrapper(query, k: int = 5):
                t0 = time.perf_counter()
                out, meta = await self._orig_asearch(query, k)
                latency = time.perf_counter() - t0
                self.log.append({
                    "time": time.time(),
                    "hit": float(bool(meta)),
                    "latency": latency,
                })
                return out, meta

            self.memory.asearch = asearch_wrapper

    # --------------------------------------------------------------
    def stop(self) -> None:
        if self._orig_search is None:
            return
        self.memory.search = self._orig_search
        self._orig_search = None
        if self._orig_asearch is not None:
            self.memory.asearch = self._orig_asearch
            self._orig_asearch = None

    # --------------------------------------------------------------
    def to_image(self) -> str:
        if not self.log:
            return ""
        times = np.array([e["time"] for e in self.log])
        lat = np.array([e["latency"] for e in self.log])
        count = np.arange(1, len(times) + 1)
        times = times - times[0]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3))
        ax1.plot(times, count, marker="o")
        ax1.set_ylabel("retrievals")
        ax2.plot(times, lat, marker="o")
        ax2.set_ylabel("latency (s)")
        ax2.set_xlabel("time (s)")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


__all__ = ["RetrievalVisualizer"]
