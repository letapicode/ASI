import torch
from typing import Iterable, Any, Tuple, List
import torch
from .hierarchical_memory import HierarchicalMemory
from .remote_memory import RemoteMemory


class DistributedMemory(HierarchicalMemory):
    """Hierarchical memory that replicates to remote nodes."""

    def __init__(self, *args, remotes: Iterable[str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.remotes = [RemoteMemory(addr) for addr in remotes or []]

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        super().add(x, metadata)
        for r in self.remotes:
            r.add(x, metadata)

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[Any]]:
        out, meta = super().search(query, k)
        for r in self.remotes:
            r_out, r_meta = r.search(query, k)
            if r_out.numel() > 0:
                out = torch.cat([out, r_out.to(query.device)], dim=0)
                meta.extend(r_meta)
        if out.numel() == 0:
            return out, meta
        query_vec = query.view(1, -1)
        scores = out @ query_vec.T
        idx = torch.argsort(scores.ravel(), descending=True)[:k]
        return out[idx], [meta[i] for i in idx]
