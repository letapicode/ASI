import torch
from torch import nn

from .hierarchical_memory import HierarchicalMemory
from .topk_sparse_attention import topk_sparse_attention


class LinkSlotAttention(nn.Module):
    """Retrieve top-k neighbors from ``HierarchicalMemory`` and attend to them."""

    def __init__(self, memory: HierarchicalMemory, dim: int, k_top: int = 4) -> None:
        super().__init__()
        self.memory = memory
        self.dim = dim
        self.k_top = k_top
        self.query_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Augment sequence ``x`` with retrieval attention."""
        batch, seq, _ = x.shape
        outputs = []
        for t in range(seq):
            q = self.query_proj(x[:, t])
            retrieved, _ = self.memory.search(q.detach(), k=self.k_top)
            if retrieved.numel() == 0:
                out = x[:, t]
            else:
                mem = retrieved.unsqueeze(0).expand(batch, -1, -1)
                k_top = min(self.k_top, mem.size(1))
                out = topk_sparse_attention(q.unsqueeze(1), mem, mem, k_top).squeeze(1)
            self.memory.add(q.detach())
            outputs.append(out)
        return torch.stack(outputs, dim=1)
