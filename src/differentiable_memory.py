from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Any, List, Tuple

from .hierarchical_memory import HierarchicalMemory


class DifferentiableMemory(nn.Module):
    """Wrap :class:`HierarchicalMemory` with autograd support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.memory = HierarchicalMemory(*args, **kwargs)
        self.vectors: nn.ParameterList = nn.ParameterList()
        self.meta: List[Any] = []

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Store ``x`` and record parameters for gradient updates."""
        self.memory.add(x, metadata)
        metas = list(metadata) if metadata is not None else [None] * len(x)
        for row, m in zip(x, metas):
            p = nn.Parameter(row.clone().detach())
            self.vectors.append(p)
            self.meta.append(m)

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[Any]]:
        """Retrieve top-k vectors with gradients enabled."""
        if len(self.vectors) == 0:
            empty = torch.empty(0, query.size(-1), device=query.device)
            return empty, []
        mat = torch.stack([p for p in self.vectors]).to(query.device)
        scores = torch.matmul(mat, query.view(-1))
        k = min(k, len(self.vectors))
        idx = torch.topk(scores, k).indices
        out = mat[idx]
        out_meta = [self.meta[i] for i in idx.tolist()]
        return out, out_meta

    def __len__(self) -> int:
        return len(self.vectors)


__all__ = ["DifferentiableMemory"]
