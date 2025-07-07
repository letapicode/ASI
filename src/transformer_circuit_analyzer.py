from __future__ import annotations

try:  # pragma: no cover - optional heavy dep
    import torch
    import torch.nn as nn
    from .transformer_circuits import head_importance as _ablation_importance
except Exception:  # pragma: no cover - fallback when torch is missing
    torch = None  # type: ignore
    nn = None  # type: ignore

    def _ablation_importance(*_a, **_kw):  # type: ignore
        raise ImportError("torch is required for head importance")


def gradient_head_importance(model: nn.Module, x: torch.Tensor, name: str) -> torch.Tensor:
    """Return gradient-based importance scores for each head in ``name``."""
    if torch is None or nn is None:  # pragma: no cover - torch optional
        raise ImportError("torch is required for gradient_head_importance")
    attn = dict(model.named_modules()).get(name)
    if not isinstance(attn, nn.MultiheadAttention):
        raise ValueError(f"{name} is not MultiheadAttention")

    model.zero_grad()
    out = model(x)
    out.sum().backward()
    grad = attn.in_proj_weight.grad
    if grad is None:
        raise RuntimeError("no gradients recorded")

    hdim = attn.head_dim
    imps = []
    for h in range(attn.num_heads):
        start = h * hdim
        end = start + hdim
        qg = grad[start:end]
        kg = grad[attn.embed_dim + start : attn.embed_dim + end]
        vg = grad[2 * attn.embed_dim + start : 2 * attn.embed_dim + end]
        imps.append((qg.norm() + kg.norm() + vg.norm()).item())
    model.zero_grad()
    return torch.tensor(imps)


class TransformerCircuitAnalyzer:
    """Compute attention head importance via gradient or ablation."""

    def __init__(self, model: nn.Module, layer: str) -> None:
        self.model = model
        self.layer = layer

    def head_importance(
        self, x: torch.Tensor, method: str = "gradient"
    ) -> torch.Tensor:
        if torch is None or nn is None:
            raise ImportError("torch is required for head importance")
        if method == "gradient":
            return gradient_head_importance(self.model, x, self.layer)
        if method == "ablation":
            return _ablation_importance(self.model, x, self.layer)
        raise ValueError("method must be 'gradient' or 'ablation'")


__all__ = ["TransformerCircuitAnalyzer", "gradient_head_importance"]
