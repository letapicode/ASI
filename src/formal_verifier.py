import torch
from torch import nn
from typing import Iterable


def gradient_norm(module: nn.Module, inputs: torch.Tensor) -> float:
    inputs = inputs.clone().detach().requires_grad_(True)
    out = module(inputs)
    loss = out.pow(2).mean()
    loss.backward()
    grad_norm = inputs.grad.norm().item()
    return grad_norm


def verify_model(
    module: nn.Module,
    sample_inputs: Iterable[torch.Tensor],
    output_bounds: tuple[float, float] = (-1.0, 1.0),
    max_grad_norm: float = 5.0,
) -> bool:
    """Return ``True`` if module passes basic property checks."""
    for inp in sample_inputs:
        with torch.no_grad():
            out = module(inp)
            if out.min() < output_bounds[0] or out.max() > output_bounds[1]:
                return False
        if gradient_norm(module, inp) > max_grad_norm:
            return False
    return True
