import numpy as np
import torch
from typing import Iterable, Any

from .hierarchical_memory import HierarchicalMemory


class DifferentialPrivacyMemory(HierarchicalMemory):
    """HierarchicalMemory that injects noise for differential privacy."""

    def __init__(self, *args, dp_epsilon: float = 1.0, noise: str = "gaussian", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dp_epsilon = float(dp_epsilon)
        self.noise = noise

    def _apply_noise(self, arr: np.ndarray) -> np.ndarray:
        scale = 1.0 / max(self.dp_epsilon, 1e-6)
        if self.noise == "laplace":
            noise = np.random.laplace(scale=scale, size=arr.shape)
        else:
            noise = np.random.normal(scale=scale, size=arr.shape)
        return arr + noise.astype(np.float32)

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Compress ``x`` and store a noisy version."""
        self.compressor.add(x)
        comp = self.compressor.encoder(x).detach().cpu().numpy()
        comp = self._apply_noise(comp)
        metas = list(metadata) if metadata is not None else []
        if not metas:
            metas = [self._next_id + i for i in range(comp.shape[0])]
            self._next_id += comp.shape[0]
        self.store.add(comp, metas)
        for m in metas:
            self._usage[m] = 0
        if self.kg is not None:
            triples = [t for t in metas if isinstance(t, tuple) and len(t) == 3]
            if triples:
                self.kg.add_triples(triples)
        self._evict_if_needed()


__all__ = ["DifferentialPrivacyMemory"]

