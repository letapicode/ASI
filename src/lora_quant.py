import math
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["LoRAQuantLinear", "apply_quant_lora"]


def _quantize_4bit(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to int4 with symmetric scaling.

    Returns the quantized values (stored in int8) and the scale factor.
    """
    scale = t.abs().max() / 7.0 + 1e-8
    q = torch.round(t / scale).clamp(-8, 7).to(torch.int8)
    return q, scale


def _dequantize_4bit(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


class LoRAQuantLinear(nn.Module):
    """Linear layer with a 4-bit LoRA adapter."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if r > 0:
            self.lora_a = nn.Parameter(torch.zeros(r, base.in_features))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)
        # buffers for quantised weights when not training
        self.register_buffer("qa", None)
        self.register_buffer("qb", None)
        self.register_buffer("scale_a", None)
        self.register_buffer("scale_b", None)

    def quantize(self):
        """Quantize LoRA weights and store them as buffers."""
        if self.r == 0:
            return
        self.qa, self.scale_a = _quantize_4bit(self.lora_a.detach())
        self.qb, self.scale_b = _quantize_4bit(self.lora_b.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r == 0:
            return out
        if self.training or self.qa is None:
            a = self.lora_a
            b = self.lora_b
        else:
            a = _dequantize_4bit(self.qa, self.scale_a)
            b = _dequantize_4bit(self.qb, self.scale_b)
        if self.dropout is not None:
            x = self.dropout(x)
        lora_out = F.linear(x, b @ a, bias=None) * self.scale
        return out + lora_out


def apply_quant_lora(model: nn.Module, target_modules: Sequence[str], r: int = 4,
                     alpha: float = 1.0, dropout: float = 0.0) -> nn.Module:
    """Wrap ``target_modules`` in ``model`` with ``LoRAQuantLinear``.

    Parameters
    ----------
    model:
        The model whose modules will be replaced in-place.
    target_modules:
        Iterable of attribute names (e.g. ``["q_proj", "v_proj"]``) to adapt.
    r:
        Rank of the LoRA adapters.
    alpha:
        LoRA scaling factor.
    dropout:
        Optional dropout probability for the injected adapters.
    """
    modules = list(model.named_modules())
    for name, module in modules:
        for tgt in target_modules:
            if name.split(".")[-1] == tgt and isinstance(module, nn.Linear):
                parent = model
                for attr in name.split(".")[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, tgt, LoRAQuantLinear(module, r=r, alpha=alpha, dropout=dropout))
    return model
