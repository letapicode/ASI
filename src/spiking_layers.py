from __future__ import annotations

import torch
from torch import nn


class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(input)
        return (input >= threshold).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (input,) = ctx.saved_tensors
        sg = torch.sigmoid(input) * (1 - torch.sigmoid(input))
        return grad_output * sg, None


class LIFNeuron(nn.Module):
    """Simplified leaky integrate-and-fire neuron."""

    def __init__(self, decay: float = 0.9, threshold: float = 1.0) -> None:
        super().__init__()
        self.decay = decay
        self.threshold = threshold
        self.register_buffer("mem", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.mem.numel() != x.numel() or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        self.mem = self.mem * self.decay + x
        spk = _SpikeFn.apply(self.mem, self.threshold)
        self.mem = self.mem - spk * self.threshold
        return spk

    def reset_state(self) -> None:
        self.mem.zero_()


class SpikingLinear(nn.Module):
    """Linear layer followed by an LIF neuron."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        decay: float = 0.9,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.neuron = LIFNeuron(decay=decay, threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.neuron(self.linear(x))

    def reset_state(self) -> None:
        self.neuron.reset_state()


__all__ = ["LIFNeuron", "SpikingLinear"]
