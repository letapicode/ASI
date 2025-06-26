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
        """Persist compressor state and vector store to ``path``."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        comp_state = {
            "state_dict": self.compressor.state_dict(),
            "buffer": [t.cpu() for t in self.compressor.buffer.data],
            "count": self.compressor.buffer.count,
            "capacity": self.compressor.buffer.capacity,
        }
        torch.save(comp_state, p / "compressor.pt")
        self.store.save(p / "vectors.npz")

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalMemory":
        """Load a saved ``HierarchicalMemory`` from ``save()`` output."""
        p = Path(path)
        comp_state = torch.load(p / "compressor.pt", map_location="cpu")
        state_dict = comp_state["state_dict"]
        dim = state_dict["encoder.weight"].shape[1]
        compressed_dim = state_dict["encoder.weight"].shape[0]
        capacity = int(comp_state.get("capacity", 0))
        mem = cls(dim, compressed_dim, capacity)
        mem.compressor.load_state_dict(state_dict)
        mem.compressor.buffer.data = [t for t in comp_state["buffer"]]
        mem.compressor.buffer.count = int(comp_state["count"])
        mem.store = VectorStore.load(p / "vectors.npz")
        return mem
