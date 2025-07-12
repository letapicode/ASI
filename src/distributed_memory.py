"""Distributed hierarchical memory with remote replication."""

from __future__ import annotations

from typing import Iterable, Any, Tuple, List

import torch

from .hierarchical_memory import HierarchicalMemory
from .memory_clients import push_remote, query_remote


class DistributedMemory(HierarchicalMemory):
    """Hierarchical memory that replicates operations to remote nodes."""

    def __init__(self, *args, remotes: Iterable[str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.remotes = list(remotes or [])

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        super().add(x, metadata)
        if not self.remotes:
            return
        data = x if x.ndim == 2 else x.unsqueeze(0)
        metas = list(metadata) if metadata is not None else [None] * len(data)
        for addr in self.remotes:
            for vec, meta in zip(data, metas):
                push_remote(addr, vec, meta)

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[Any]]:
        out, meta = super().search(query, k)
        if self.remotes:
            for addr in self.remotes:
                r_out, r_meta = query_remote(addr, query, k)
                if r_out.numel() > 0:
                    out = torch.cat([out, r_out.to(query.device)], dim=0)
                    meta.extend(r_meta)
        if out.numel() == 0:
            return out, meta
        scores = out @ query.view(-1, 1)
        idx = torch.argsort(scores.ravel(), descending=True)[:k]
        return out[idx], [meta[i] for i in idx]


__all__ = ["DistributedMemory"]
