from __future__ import annotations

import torch
from torch import Tensor


def token_saliency(query: Tensor, results: Tensor) -> Tensor:
    """Return saliency scores per query token for each retrieved result."""
    q = query.detach()
    if q.ndim == 1:
        q = q.unsqueeze(0)
    q = q.clone().requires_grad_(True)
    sal = []
    for r in results:
        score = (q.mean(dim=0) * r).sum()
        grad, = torch.autograd.grad(score, q, retain_graph=True)
        sal.append(grad.abs().sum(dim=-1))
    return torch.stack(sal)


def image_saliency(image: Tensor, results: Tensor) -> Tensor:
    """Return saliency maps over image patches for each retrieved result."""
    q = image.clone().detach().requires_grad_(True)
    flat = q.view(-1)
    sal = []
    for r in results:
        score = (flat * r.view(-1)).sum()
        grad, = torch.autograd.grad(score, q, retain_graph=True)
        sal.append(grad.abs().sum(dim=0))
    return torch.stack(sal)


__all__ = ["token_saliency", "image_saliency"]
