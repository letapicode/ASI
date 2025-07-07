from __future__ import annotations

from typing import Any, List
import numpy as np
import torch

from .telemetry import TelemetryLogger


class MemoryPruningManager:
    """Monitor memory usage and prune rarely accessed vectors."""

    def __init__(
        self,
        threshold: int = 1,
        summarizer: Any | None = None,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self.threshold = int(threshold)
        self.summarizer = summarizer
        self.telemetry = telemetry
        self.memory: "HierarchicalMemory | None" = None

    # --------------------------------------------------------------
    def attach(self, memory: "HierarchicalMemory") -> None:
        """Attach to a :class:`HierarchicalMemory` instance."""
        self.memory = memory

    # --------------------------------------------------------------
    def _get_vector(self, meta: Any) -> np.ndarray | None:
        if self.memory is None:
            return None
        store = self.memory.store
        mlist = getattr(store, "_meta", getattr(store, "meta", []))
        try:
            idx = mlist.index(meta)
        except ValueError:
            return None
        vecs = getattr(store, "_vectors", getattr(store, "vectors", None))
        if vecs is None:
            return None
        if isinstance(vecs, list):
            if not vecs:
                return None
            mat = np.concatenate(vecs, axis=0)
        else:
            mat = vecs
        if idx >= len(mat):
            return None
        return mat[idx]

    # --------------------------------------------------------------
    def prune(self) -> None:
        """Prune or summarize entries below the usage ``threshold``."""
        if self.memory is None:
            return
        removed: List[Any] = []
        for meta, count in list(self.memory._usage.items()):
            if count >= self.threshold:
                continue
            vec = self._get_vector(meta)
            if self.summarizer is not None and vec is not None:
                comp = torch.from_numpy(vec).unsqueeze(0)
                full = self.memory.compressor.decoder(comp).squeeze(0)
                if hasattr(self.summarizer, "summarize"):
                    text = self.summarizer.summarize(full)
                else:
                    text = self.summarizer(full)
                self.memory.store.delete(tag=meta)
                zero = np.zeros((1, vec.shape[-1]), dtype=np.float32)
                self.memory.store.add(zero, [{"summary": text}])
            else:
                self.memory.store.delete(tag=meta)
            self.memory._usage.pop(meta, None)
            removed.append(meta)
        if removed and self.telemetry is not None:
            for m in removed:
                self.telemetry.events.append({"event": "memory_prune", "meta": str(m)})


__all__ = ["MemoryPruningManager"]

