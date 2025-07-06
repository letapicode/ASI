from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import torch

from .retrieval_saliency import image_saliency, token_saliency


class RetrievalVisualizer:
    """Record retrieval hit/miss events and render simple plots."""

    def __init__(self, memory: Any) -> None:
        self.memory = memory
        self.log: List[Dict[str, float]] = []
        self.saliencies: List[np.ndarray] = []
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
            try:
                q = torch.tensor(query, dtype=torch.float32)
                r = torch.tensor(out, dtype=torch.float32)
                if q.ndim == 3 and r.shape[1] == q.numel():
                    sal = image_saliency(q, r)
                else:
                    sal = token_saliency(q, r)
                self.saliencies.append(sal.detach().cpu().numpy())
            except Exception:
                pass
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
                try:
                    q = torch.tensor(query, dtype=torch.float32)
                    r = torch.tensor(out, dtype=torch.float32)
                    if q.ndim == 3 and r.shape[1] == q.numel():
                        sal = image_saliency(q, r)
                    else:
                        sal = token_saliency(q, r)
                    self.saliencies.append(sal.detach().cpu().numpy())
                except Exception:
                    pass
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

    # --------------------------------------------------------------
    def pattern_image(self) -> str:
        if not self.log:
            return ""
        times = np.array([e["time"] for e in self.log])
        lat = np.array([e["latency"] for e in self.log])
        count = np.arange(1, len(times) + 1)
        times = times - times[0]
        nrows = 3 if self.saliencies else 2
        fig, axes = plt.subplots(nrows, 1, figsize=(4, 3 + (1 if self.saliencies else 0)))
        ax1, ax2 = axes[0], axes[1]
        ax1.plot(times, count, marker="o")
        ax1.set_ylabel("retrievals")
        ax2.plot(times, lat, marker="o")
        ax2.set_ylabel("latency (s)")
        ax2.set_xlabel("time (s)")
        if self.saliencies:
            ax3 = axes[2]
            sal = self.saliencies[-1][0]
            if sal.ndim == 1:
                ax3.imshow(sal[None, :], aspect="auto", cmap="hot")
            else:
                ax3.imshow(sal, aspect="auto", cmap="hot")
            ax3.set_ylabel("saliency")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


__all__ = ["RetrievalVisualizer"]
