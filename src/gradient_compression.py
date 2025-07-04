import torch
from dataclasses import dataclass

__all__ = ["GradientCompressionConfig", "GradientCompressor"]


@dataclass
class GradientCompressionConfig:
    """Configuration for gradient compression."""

    topk: int | None = None
    bits: int | None = None


class GradientCompressor:
    """Apply top-k sparsification or quantization to gradients."""

    def __init__(self, cfg: GradientCompressionConfig) -> None:
        self.cfg = cfg

    def _topk(self, g: torch.Tensor) -> torch.Tensor:
        k = self.cfg.topk
        if k is None or k <= 0 or k >= g.numel():
            return g
        flat = g.view(-1)
        _, idx = torch.topk(flat.abs(), k)
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        return (flat * mask).view_as(g)

    def _quantize(self, g: torch.Tensor) -> torch.Tensor:
        bits = self.cfg.bits or 8
        qmax = 2 ** (bits - 1) - 1
        scale = g.abs().max() / qmax + 1e-8
        q = torch.round(g / scale).clamp(-qmax - 1, qmax).to(torch.int8)
        return q.float() * scale

    def compress(self, grad: torch.Tensor) -> torch.Tensor:
        out = grad
        if self.cfg.topk is not None:
            out = self._topk(out)
        if self.cfg.bits is not None:
            out = self._quantize(out)
        return out

    def compress_dict(self, grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply compression to all tensors in a gradient dict."""
        return {k: self.compress(v) for k, v in grads.items()}
