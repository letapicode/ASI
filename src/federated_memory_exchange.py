from __future__ import annotations

from typing import Iterable, Any, Tuple, List

import torch

from .hierarchical_memory import HierarchicalMemory
from .remote_memory import push_batch_remote, query_remote


class FederatedMemoryExchange:
    """Replicate memory operations across multiple ``MemoryServer`` nodes."""

    def __init__(
        self,
        local_memory: HierarchicalMemory,
        peers: Iterable[str] | None = None,
    ) -> None:
        self.local = local_memory
        self.peers = list(peers or [])

    def add_peer(self, address: str) -> None:
        """Register a new peer address."""
        if address not in self.peers:
            self.peers.append(address)

    def push(
        self, vectors: torch.Tensor, metadata: Iterable[Any] | None = None
    ) -> None:
        """Store ``vectors`` locally and replicate them to all peers."""
        self.local.add(vectors, metadata)
        if not self.peers:
            return
        for addr in self.peers:
            push_batch_remote(addr, vectors, metadata)

    def query(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[Any]]:
        """Query ``k`` nearest vectors across all peers and the local store."""
        out, meta = self.local.search(query, k)
        for addr in self.peers:
            r_vec, r_meta = query_remote(addr, query, k)
            if r_vec.numel() > 0:
                out = torch.cat([out, r_vec.to(query.device)], dim=0)
                meta.extend(r_meta)
        if out.numel() == 0:
            return out, meta
        scores = out @ query.view(-1, 1)
        idx = torch.argsort(scores.ravel(), descending=True)[:k]
        return out[idx], [meta[i] for i in idx]


__all__ = ["FederatedMemoryExchange"]
