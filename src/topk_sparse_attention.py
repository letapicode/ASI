import torch
from torch import nn
from typing import Tuple


def topk_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_top: int) -> torch.Tensor:
    """Top-k sparse attention used for Plan.md C-5.

    Args:
        q: Query tensor of shape ``(batch, seq_q, dim)``.
        k: Key tensor of shape ``(batch, seq_k, dim)``.
        v: Value tensor of shape ``(batch, seq_k, dim)``.
        k_top: Number of keys to attend per query.

    Returns:
        Tensor of shape ``(batch, seq_q, dim)`` with attended outputs.
    """
    dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / dim ** 0.5
    topk_scores, indices = torch.topk(scores, k_top, dim=-1)

    expanded_v = v.unsqueeze(1).expand(-1, q.size(1), -1, -1)
    gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, dim)
    topk_v = expanded_v.gather(2, gather_idx)

    attn = torch.softmax(topk_scores, dim=-1)
    out = (attn.unsqueeze(-1) * topk_v).sum(dim=2)
    return out
