import numpy as np
import torch
from typing import Iterable, Any, Tuple, List
from pathlib import Path

from .streaming_compression import StreamingCompressor
from .vector_store import VectorStore


class HierarchicalMemory:
    """Combine streaming compression with a vector store."""

    def __init__(self, dim: int, compressed_dim: int, capacity: int) -> None:
        self.compressor = StreamingCompressor(dim, compressed_dim, capacity)
        self.store = VectorStore(dim=compressed_dim)

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Compress and store embeddings with optional metadata."""
        self.compressor.add(x)
        comp = self.compressor.encoder(x).detach().cpu().numpy()
        self.store.add(comp, metadata)

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[Any]]:
        """Retrieve top-k decoded vectors and their metadata."""
        q = self.compressor.encoder(query).detach().cpu().numpy()
        if q.ndim == 2:
            q = q[0]
        comp_vecs, meta = self.store.search(q, k)
        if comp_vecs.shape[0] == 0:
            return torch.empty(0, query.size(-1)), meta
        comp_t = torch.from_numpy(comp_vecs)
        decoded = self.compressor.decoder(comp_t)
        return decoded, meta

    def save(self, path: str | Path) -> None:
        """Persist compressor and vector store state."""
        base = Path(path)
        comp_path = base.with_suffix("_compressor.pt")
        store_path = base.with_suffix("_store.npz")

        # Save vector store
        self.store.save(store_path)

        # Prepare compressor payload
        payload = {
            "dim": self.compressor.encoder.in_features,
            "compressed_dim": self.compressor.encoder.out_features,
            "capacity": self.compressor.buffer.capacity,
            "state_dict": self.compressor.state_dict(),
            "buffer_data": [t.detach().cpu() for t in self.compressor.buffer.data],
            "buffer_count": self.compressor.buffer.count,
        }
        torch.save(payload, comp_path)

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalMemory":
        """Load ``HierarchicalMemory`` from ``save()`` output."""
        base = Path(path)
        comp_path = base.with_suffix("_compressor.pt")
        store_path = base.with_suffix("_store.npz")

        comp_data = torch.load(comp_path, map_location="cpu")
        mem = cls(
            dim=int(comp_data["dim"]),
            compressed_dim=int(comp_data["compressed_dim"]),
            capacity=int(comp_data["capacity"]),
        )
        mem.store = VectorStore.load(store_path)
        mem.compressor.load_state_dict(comp_data["state_dict"])
        mem.compressor.buffer.data = list(comp_data["buffer_data"])
        mem.compressor.buffer.count = int(comp_data["buffer_count"])
        return mem
