import numpy as np

try:  # optional torch dependency
    import torch
except Exception:  # pragma: no cover - allow running without torch
    import types

    class _DummyTensor:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=np.float32)

        def dim(self) -> int:
            return self.data.ndim

        def unsqueeze(self, axis: int):
            return _DummyTensor(np.expand_dims(self.data, axis))

        def expand_as(self, other):
            arr = self.data
            if arr.shape[-1] != other.data.shape[-1]:
                arr = arr[..., : other.data.shape[-1]]
            return _DummyTensor(np.broadcast_to(arr, other.data.shape))

        def clone(self):
            return _DummyTensor(self.data.copy())

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

        def __iter__(self):
            if self.data.ndim == 0:
                yield _DummyTensor(self.data)
            else:
                for row in self.data:
                    yield _DummyTensor(row)

        def size(self, dim=None):
            return self.data.shape if dim is None else self.data.shape[dim]

        def to(self, *args, **kwargs):
            return self

        @property
        def device(self):
            return "cpu"

        def numel(self):
            return self.data.size

        def tolist(self):
            return self.data.tolist()

        def item(self):
            return float(self.data)

    class _DummyTorch(types.SimpleNamespace):
        Tensor = _DummyTensor

        def from_numpy(self, arr):
            return _DummyTensor(arr)

        def stack(self, seq):
            return _DummyTensor(np.stack([s.data for s in seq]))

    torch = _DummyTorch()

_BASE_VECS = {
    "man": torch.tensor([1.0, 0.0, 0.0]),
    "woman": torch.tensor([0.0, 1.0, 0.0]),
    "king": torch.tensor([1.0, 0.0, 1.0]),
    "queen": torch.tensor([0.0, 1.0, 1.0]),
    "france": torch.tensor([1.0, 0.0, 0.0]),
    "germany": torch.tensor([0.0, 1.0, 0.0]),
    "paris": torch.tensor([1.0, 0.0, 1.0]),
    "berlin": torch.tensor([0.0, 1.0, 1.0]),
}

def embed_text(text: str, dim: int):
    """Deterministically embed ``text`` with simple seed vectors."""
    base = text.split(" ")[-1]
    if base in _BASE_VECS:
        vec = _BASE_VECS[base]
    else:
        seed = abs(hash(text)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        vec = torch.from_numpy(rng.standard_normal(dim).astype(np.float32))
    if vec.numel() != dim:
        vec = torch.nn.functional.pad(vec, (0, dim - vec.numel()))
    return vec

__all__ = ["embed_text"]
