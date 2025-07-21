# Fallback utilities mimicking minimal torch functionality

from __future__ import annotations

import types
from typing import Iterable, Any
import numpy as np


class DummyTensor:
    """Lightweight stand-in for ``torch.Tensor`` when torch is unavailable."""

    def __init__(self, data: Any):
        self.data = np.asarray(data, dtype=np.float32)

    # tensor-like helpers -------------------------------------------------
    def dim(self) -> int:
        return self.data.ndim

    def unsqueeze(self, axis: int) -> "DummyTensor":
        return DummyTensor(np.expand_dims(self.data, axis))

    def expand_as(self, other: "DummyTensor") -> "DummyTensor":
        arr = self.data
        if arr.shape[-1] != other.data.shape[-1]:
            arr = arr[..., : other.data.shape[-1]]
        return DummyTensor(np.broadcast_to(arr, other.data.shape))

    def clone(self) -> "DummyTensor":
        return DummyTensor(self.data.copy())

    def detach(self) -> "DummyTensor":
        return self

    def cpu(self) -> "DummyTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.data

    def __getitem__(self, idx: int) -> "DummyTensor":
        return DummyTensor(self.data[idx])

    def __iter__(self):
        if self.data.ndim == 0:
            yield DummyTensor(self.data)
        else:
            for row in self.data:
                yield DummyTensor(row)

    def size(self, dim: int | None = None):
        return self.data.shape if dim is None else self.data.shape[dim]

    def to(self, *_args: Any, **_kw: Any) -> "DummyTensor":
        return self

    @property
    def device(self) -> str:
        return "cpu"

    def numel(self) -> int:
        return self.data.size

    def tolist(self) -> list:
        return self.data.tolist()

    def item(self):
        return float(self.data)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def view(self, *shape: int) -> "DummyTensor":
        return DummyTensor(np.reshape(self.data, shape))

    reshape = view


class DummyNN(types.SimpleNamespace):
    class Module:  # type: ignore[override]
        pass

    class Linear:
        def __init__(self, in_f: int, out_f: int):
            self.in_features = in_f
            self.out_features = out_f

        def __call__(self, x: Any) -> DummyTensor:
            arr = getattr(x, "data", np.asarray(x, dtype=np.float32))
            out = arr[..., : self.out_features]
            return DummyTensor(out)

    class functional(types.SimpleNamespace):
        @staticmethod
        def cosine_similarity(a: Any, b: Any, dim: int = 1) -> DummyTensor:
            a = getattr(a, "data", a)
            b = getattr(b, "data", b)
            dot = (a * b).sum(axis=dim)
            na = np.linalg.norm(a, axis=dim)
            nb = np.linalg.norm(b, axis=dim)
            return DummyTensor(dot / (na * nb + 1e-8))


class DummyTorch(types.SimpleNamespace):
    Tensor = DummyTensor
    float32 = "float32"

    def empty(self, *shape: int, device: Any | None = None):
        return DummyTensor(np.empty(shape, dtype=np.float32))

    def stack(self, seq: Iterable[Any], dim: int = 0):
        return DummyTensor(np.stack([getattr(s, "data", s) for s in seq], axis=dim))

    def from_numpy(self, arr: Any):
        return DummyTensor(np.asarray(arr, dtype=np.float32))

    def cat(self, seq: Iterable[DummyTensor], dim: int = 0):
        return DummyTensor(np.concatenate([s.data for s in seq], axis=dim))

    def tensor(self, data: Any):
        return DummyTensor(np.asarray(data, dtype=np.float32))

    def randn(self, *shape: int):
        return DummyTensor(np.random.randn(*shape).astype(np.float32))


# instantiate global objects matching torch API
nn = DummyNN()
torch = DummyTorch(nn=nn)

__all__ = ["torch", "nn", "DummyTensor"]
