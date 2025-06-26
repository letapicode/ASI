import numpy as np
import torch
from pathlib import Path
from typing import Iterable, Any, Tuple, List

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
            empty = torch.empty(0, query.size(-1), device=query.device)
            return empty, meta
        comp_t = torch.from_numpy(comp_vecs)
        decoded = self.compressor.decoder(comp_t)
        return decoded.to(query.device), meta

    def save(self, path: str | Path) -> None:
        """Persist compressor state and vector store to ``path``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        comp_state = {
            "dim": self.compressor.encoder.in_features,
            "compressed_dim": self.compressor.encoder.out_features,
            "capacity": self.compressor.buffer.capacity,
            "buffer": [t.cpu() for t in self.compressor.buffer.data],
            "count": self.compressor.buffer.count,
            "encoder": self.compressor.encoder.state_dict(),
            "decoder": self.compressor.decoder.state_dict(),
        }
        torch.save(comp_state, path / "compressor.pt")
        self.store.save(path / "store.npz")

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalMemory":
        """Load memory from ``save()`` output."""
        path = Path(path)
        state = torch.load(path / "compressor.pt", map_location="cpu")
        mem = cls(
            dim=int(state["dim"]),
            compressed_dim=int(state["compressed_dim"]),
            capacity=int(state["capacity"]),
        )
        mem.compressor.encoder.load_state_dict(state["encoder"])
        mem.compressor.decoder.load_state_dict(state["decoder"])
        mem.compressor.buffer.data = [t.clone() for t in state["buffer"]]
        mem.compressor.buffer.count = int(state["count"])
        mem.store = VectorStore.load(path / "store.npz")
        return mem
