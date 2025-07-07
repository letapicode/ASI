"""Graph Neural Network memory over GraphOfThought nodes."""

from __future__ import annotations

import random
from typing import Iterable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .graph_of_thought import ThoughtNode


def _embed_text(text: str, dim: int) -> torch.Tensor:
    """Deterministically embed ``text`` using a hash based RNG."""
    seed = abs(hash(text)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return torch.from_numpy(vec)


class GNNMemory(nn.Module):
    """Simple GraphSAGE-style encoder for reasoning graphs."""

    def __init__(
        self,
        nodes: Iterable[ThoughtNode],
        edges: Dict[int, List[int]],
        dim: int = 16,
    ) -> None:
        super().__init__()
        node_list = list(nodes)
        self.id_to_idx = {n.id: i for i, n in enumerate(node_list)}
        self.idx_to_id = {i: n.id for i, n in enumerate(node_list)}
        self.edges = {k: list(v) for k, v in edges.items()}
        self.dim = dim
        self.embed = nn.Embedding(len(node_list), dim)
        with torch.no_grad():
            init = torch.stack([_embed_text(n.text, dim) for n in node_list])
            self.embed.weight.copy_(init)
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)

    # ------------------------------------------------------------------
    def encode_nodes(self) -> torch.Tensor:
        """Return embeddings for all nodes after one message pass."""
        h = self.embed.weight
        neigh = torch.zeros_like(h)
        for src, dsts in self.edges.items():
            if src not in self.id_to_idx or not dsts:
                continue
            idx = self.id_to_idx[src]
            idxs = [self.id_to_idx[d] for d in dsts if d in self.id_to_idx]
            if idxs:
                neigh[idx] = h[idxs].mean(dim=0)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        return F.relu(out)

    # ------------------------------------------------------------------
    def edge_loss(self) -> torch.Tensor:
        """Return link prediction loss over observed edges."""
        h = self.encode_nodes()
        n = h.size(0)
        loss = 0.0
        count = 0
        for src, dsts in self.edges.items():
            src_idx = self.id_to_idx.get(src)
            if src_idx is None:
                continue
            for dst in dsts:
                dst_idx = self.id_to_idx.get(dst)
                if dst_idx is None:
                    continue
                pos_score = (h[src_idx] * h[dst_idx]).sum()
                neg = random.randrange(n)
                neg_score = (h[src_idx] * h[neg]).sum()
                loss += F.binary_cross_entropy_with_logits(pos_score, torch.tensor(1.0))
                loss += F.binary_cross_entropy_with_logits(neg_score, torch.tensor(0.0))
                count += 2
        if count == 0:
            return h.sum() * 0
        return loss / count

    # ------------------------------------------------------------------
    def query(self, context: int | Iterable[int]) -> Tuple[torch.Tensor, List[int]]:
        """Return neighbour embeddings for ``context`` node(s)."""
        if isinstance(context, int):
            ctx = [context]
        else:
            ctx = list(context)
        neigh_ids = set()
        for n in ctx:
            neigh_ids.update(self.edges.get(n, []))
        idxs = [self.id_to_idx[i] for i in neigh_ids if i in self.id_to_idx]
        if not idxs:
            return torch.empty(0, self.dim), []
        emb = self.encode_nodes()
        return emb[idxs], [self.idx_to_id[i] for i in idxs]


__all__ = ["GNNMemory"]
