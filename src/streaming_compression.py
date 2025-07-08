import random
from typing import List, Any

__all__ = ["ReservoirBuffer", "StreamingCompressor"]

try:  # optional torch dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - allow running without torch
    import types
    import numpy as np

    class _DummyNN(types.SimpleNamespace):
        class Module:  # type: ignore[override]
            pass

        class Linear:
            def __init__(self, in_f: int, out_f: int):
                self.in_features = in_f
                self.out_features = out_f
            def __call__(self, x):
                return x

    class _DummyTorch(types.SimpleNamespace):
        Tensor = type("Tensor", (), {})

        def empty(self, *args: Any):
            return np.empty(*args)

        def stack(self, seq):
            return np.stack(list(seq))

    torch = _DummyTorch()
    nn = _DummyNN()


class ReservoirBuffer:
    """Maintain a reservoir sample of token embeddings."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: List[torch.Tensor] = []
        self.count = 0

    def add(self, x: torch.Tensor) -> None:
        """Add a batch of embeddings to the reservoir."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for row in x:
            self.count += 1
            if len(self.data) < self.capacity:
                self.data.append(row.detach().clone())
            else:
                j = random.randint(0, self.count - 1)
                if j < self.capacity:
                    self.data[j] = row.detach().clone()

    def get(self) -> torch.Tensor:
        if not self.data:
            return torch.empty(0)
        return torch.stack(self.data)


class StreamingCompressor(nn.Module):
    """Reservoir sampling with a small autoencoder for lossy storage."""

    def __init__(self, dim: int, compressed_dim: int, capacity: int) -> None:
        super().__init__()
        self.buffer = ReservoirBuffer(capacity)
        self.encoder = nn.Linear(dim, compressed_dim)
        self.decoder = nn.Linear(compressed_dim, dim)

    def add(self, x: torch.Tensor) -> None:
        """Store new embeddings in the reservoir."""
        self.buffer.add(x)

    def compressed(self) -> torch.Tensor:
        """Return compressed representations for the buffered data."""
        data = self.buffer.get()
        if data.numel() == 0:
            return data
        return self.encoder(data)

    def reconstruct(self) -> torch.Tensor:
        """Decode the compressed reservoir back to the original dimension."""
        comp = self.compressed()
        if comp.numel() == 0:
            return comp
        return self.decoder(comp)


class AdaptiveCompressor(StreamingCompressor):
    """Adjust compression ratio based on retrieval frequency."""

    def __init__(self, dim: int, compressed_dim: int, capacity: int, min_dim: int = 4) -> None:
        super().__init__(dim, compressed_dim, capacity)
        self.min_dim = min_dim
        self.hits = 0
        self.misses = 0

    def update_ratio(self) -> None:
        total = self.hits + self.misses
        if total < 10:
            return
        hit_rate = self.hits / total
        target = int(self.encoder.out_features * (0.5 + 0.5 * hit_rate))
        target = max(self.min_dim, min(target, self.encoder.in_features))
        if target != self.encoder.out_features:
            self.encoder = nn.Linear(self.encoder.in_features, target)
            self.decoder = nn.Linear(target, self.decoder.out_features)
        self.hits = 0
        self.misses = 0

    def record_hit(self, hit: bool) -> None:
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        self.update_ratio()


__all__.append("AdaptiveCompressor")


class TemporalVectorCompressor(StreamingCompressor):
    """Compressor that exponentially decays old vectors."""

    def __init__(
        self, dim: int, compressed_dim: int, capacity: int, decay: float = 0.99
    ) -> None:
        super().__init__(dim, compressed_dim, capacity)
        self.decay = decay
        self.weights: List[float] = []

    def add(self, x: torch.Tensor) -> None:  # type: ignore[override]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for row in x:
            self.buffer.count += 1
            self.weights = [w * self.decay for w in self.weights]
            if len(self.buffer.data) < self.buffer.capacity:
                self.buffer.data.append(row.detach().clone())
                self.weights.append(1.0)
            else:
                self.buffer.data.append(row.detach().clone())
                self.weights.append(1.0)
                idx = self.weights.index(min(self.weights))
                self.buffer.data.pop(idx)
                self.weights.pop(idx)


__all__.extend(["TemporalVectorCompressor"])
