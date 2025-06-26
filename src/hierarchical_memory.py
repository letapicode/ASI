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
            return torch.empty(0, query.size(-1)), meta
        comp_t = torch.from_numpy(comp_vecs)
        decoded = self.compressor.decoder(comp_t)
        return decoded, meta

    def save(self, path: str | Path) -> None:
        """Persist compressor state and vector store to ``path``."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        state = {
            "dim": self.compressor.encoder.in_features,
            "compressed_dim": self.compressor.encoder.out_features,
            "capacity": self.compressor.buffer.capacity,
            "buffer_data": [t.detach().cpu() for t in self.compressor.buffer.data],
            "buffer_count": self.compressor.buffer.count,
            "encoder": self.compressor.encoder.state_dict(),
            "decoder": self.compressor.decoder.state_dict(),
        }
        torch.save(state, p / "compressor.pt")
        self.store.save(p / "store.npz")

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalMemory":
        """Load from ``save()`` output and return a new instance."""
        p = Path(path)
        state = torch.load(p / "compressor.pt", map_location="cpu")
        mem = cls(state["dim"], state["compressed_dim"], state["capacity"])
        mem.compressor.encoder.load_state_dict(state["encoder"])
        mem.compressor.decoder.load_state_dict(state["decoder"])
        mem.compressor.buffer.data = [t.clone() for t in state["buffer_data"]]
        mem.compressor.buffer.count = state["buffer_count"]
        mem.store = VectorStore.load(p / "store.npz")
        return mem
