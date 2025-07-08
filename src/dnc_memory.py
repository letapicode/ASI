"""A lightweight Differentiable Neural Computer memory backend."""

from __future__ import annotations

import numpy as np
from typing import Iterable, Any, List, Tuple
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

        def tolist(self):
            return self.data.tolist()

        def detach(self) -> "_DummyTensor":
            return self

        def cpu(self) -> "_DummyTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self.data

    class _DummyTorch(types.SimpleNamespace):
        Tensor = _DummyTensor

        float32 = 'float32'

        def empty(self, *args: Any):
            return _DummyTensor(np.empty(*args))

        def stack(self, seq):
            return _DummyTensor(np.stack([s.data for s in seq]))

        def from_numpy(self, arr):
            return _DummyTensor(np.asarray(arr, dtype=np.float32))

    torch = _DummyTorch()
from pathlib import Path


class DNCMemory:
    """Minimal Differentiable Neural Computer memory.

    This implementation keeps a fixed-size memory matrix and performs
    content-based reads using cosine similarity. Writes append new vectors
    in a ring buffer manner. Metadata is stored alongside each slot.
    """

    def __init__(
        self,
        memory_size: int,
        word_size: int,
        num_reads: int = 1,
        num_writes: int = 1,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.memory_size = int(memory_size)
        self.word_size = int(word_size)
        self.num_reads = int(num_reads)
        self.num_writes = int(num_writes)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        self.memory = torch.zeros(self.memory_size, self.word_size, device=self.device, dtype=self.dtype)
        self.read_weights = torch.zeros(self.num_reads, self.memory_size, device=self.device, dtype=self.dtype)
        self.write_weights = torch.zeros(self.num_writes, self.memory_size, device=self.device, dtype=self.dtype)
        self.meta: List[Any] = [None] * self.memory_size
        self.ptr = 0
        self.count = 0

    def __len__(self) -> int:
        return self.count

    @property
    def capacity(self) -> int:
        return self.memory_size

    def reset(self) -> None:
        """Clear all memory slots."""
        self.memory.zero_()
        self.read_weights.zero_()
        self.write_weights.zero_()
        self.meta = [None] * self.memory_size
        self.ptr = 0
        self.count = 0

    # --------------------------------------------------------------
    # Write/Read operations

    def write(self, vectors: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Store ``vectors`` sequentially in memory."""
        vecs = torch.as_tensor(vectors, dtype=self.dtype, device=self.device)
        if vecs.ndim == 1:
            vecs = vecs.unsqueeze(0)
        metas = list(metadata) if metadata is not None else [None] * vecs.size(0)
        for v, m in zip(vecs, metas):
            idx = self.ptr % self.memory_size
            self.memory[idx] = v
            self.write_weights.zero_()
            self.write_weights[0, idx] = 1.0
            self.meta[idx] = m
            self.ptr = (self.ptr + 1) % self.memory_size
            self.count = min(self.count + 1, self.memory_size)

    def _content_weights(self, key: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        sim = torch.nn.functional.cosine_similarity(key.unsqueeze(0), self.memory, dim=1)
        if self.count < self.memory_size:
            mask = torch.arange(self.memory_size, device=self.device) < self.count
            sim = sim.masked_fill(~mask, -float('inf'))
        return torch.nn.functional.softmax(beta * sim, dim=0)

    def read(self, keys: torch.Tensor, k: int = 1, beta: float = 1.0) -> Tuple[torch.Tensor, List[List[Any]]]:
        """Return up to ``k`` memory vectors matching each ``key``."""
        ks = torch.as_tensor(keys, dtype=self.dtype, device=self.device)
        if ks.ndim == 1:
            ks = ks.unsqueeze(0)

        out_vecs = []
        out_meta = []
        for i, key in enumerate(ks[: self.num_reads]):
            weights = self._content_weights(key, beta)
            self.read_weights[i] = weights
            idx = torch.topk(weights, k).indices
            vec = self.memory[idx]
            meta = [self.meta[j] for j in idx.tolist()]
            out_vecs.append(vec)
            out_meta.append(meta)

        stacked = torch.stack([v.mean(dim=0) for v in out_vecs])
        return stacked if stacked.shape[0] > 1 else stacked[0].unsqueeze(0), out_meta

    # --------------------------------------------------------------
    # Convenience wrappers for VectorStore-like usage

    def add(self, vectors: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Alias of :meth:`write`. Provided for VectorStore compatibility."""
        self.write(vectors, metadata)

    def search(self, query: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, List[Any]]:
        vecs, metas = self.read(query, k=k)
        return vecs[0], metas[0]

    def delete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
        if index is None and tag is None:
            raise ValueError("index or tag must be specified")
        if index is not None:
            idxs = [int(index)] if not isinstance(index, Iterable) else [int(i) for i in index]
        else:
            idxs = [i for i, m in enumerate(self.meta) if m == tag]
        for i in idxs:
            if 0 <= i < self.memory_size and self.meta[i] is not None:
                self.memory[i].zero_()
                self.meta[i] = None
                self.count -= 1

    def save(self, path: str | Path) -> None:
        arr = self.memory.detach().cpu().numpy()
        np.savez_compressed(
            path,
            memory=arr,
            meta=np.array(self.meta, dtype=object),
            ptr=self.ptr,
            count=self.count,
            dtype=str(self.dtype),
        )

    @classmethod
    def load(cls, path: str | Path) -> 'DNCMemory':
        data = np.load(path, allow_pickle=True)
        mem = cls(
            data["memory"].shape[0],
            data["memory"].shape[1],
            device="cpu",
            dtype=getattr(torch, str(data.get("dtype", "float32"))),
        )
        mem.memory = torch.from_numpy(data["memory"]).to(mem.dtype)
        mem.meta = data["meta"].tolist()
        mem.ptr = int(data["ptr"])
        mem.count = int(data.get("count", mem.ptr))
        return mem


__all__ = ["DNCMemory"]
