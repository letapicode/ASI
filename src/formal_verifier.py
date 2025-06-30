from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable

import torch


@dataclass
class VerificationResult:
    passed: bool
    messages: list[str]


def check_grad_norm(model: torch.nn.Module, max_norm: float) -> tuple[bool, str]:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm().item()
    if total > max_norm:
        return False, f"Gradient norm {total:.2f} exceeds {max_norm}"
    return True, "grad_ok"


def check_output_bounds(output: torch.Tensor, bound: float) -> tuple[bool, str]:
    if output.abs().max().item() > bound:
        return False, "output_out_of_bounds"
    return True, "output_ok"


def verify_model(model: torch.nn.Module, checks: Iterable[Callable[[], tuple[bool, str]]]) -> VerificationResult:
    messages: list[str] = []
    for check in checks:
        ok, msg = check()
        messages.append(msg)
        if not ok:
            return VerificationResult(False, messages)
    return VerificationResult(True, messages)


__all__ = ["VerificationResult", "check_grad_norm", "check_output_bounds", "verify_model"]
