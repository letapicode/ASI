from __future__ import annotations

from typing import Any, List
import numpy as np
try:  # optional torch dependency
    import torch
except Exception:  # pragma: no cover - allow running without torch
    import types
    import numpy as np

    class _DummyTensor:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=np.float32)

        def dim(self) -> int:
            return self.data.ndim

        def unsqueeze(self, axis: int) -> "_DummyTensor":
            return _DummyTensor(np.expand_dims(self.data, axis))

        def detach(self) -> "_DummyTensor":
            return self

        def cpu(self) -> "_DummyTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self.data

        def to(self, *_args: Any, **_kw: Any) -> "_DummyTensor":
            return self

        @property
        def device(self) -> str:
            return "cpu"

        def numel(self) -> int:
            return self.data.size

        def tolist(self):
            return self.data.tolist()

        def __iter__(self):
            if self.data.ndim == 0:
                yield _DummyTensor(self.data)
            else:
                for row in self.data:
                    yield _DummyTensor(row)

        def item(self):
            return float(self.data)

        def to(self, *_args: Any, **_kw: Any) -> "_DummyTensor":
            return self

        @property
        def device(self) -> str:
            return "cpu"

        def numel(self) -> int:
            return self.data.size

        def detach(self) -> "_DummyTensor":
            return self

        def cpu(self) -> "_DummyTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self.data

    class _DummyTorch(types.SimpleNamespace):
        Tensor = _DummyTensor

        def stack(self, seq):
            return _DummyTensor(np.stack([s.data for s in seq]))

        def from_numpy(self, arr):
            return _DummyTensor(np.asarray(arr, dtype=np.float32))

    torch = _DummyTorch()

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

