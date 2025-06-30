from __future__ import annotations

from typing import Callable

import torch

__all__ = ["verify_model"]


def verify_model(model: torch.nn.Module, check: Callable[[torch.nn.Module], bool]) -> bool:
    """Run ``check`` on ``model`` and return True if it passes."""
    try:
        with torch.no_grad():
            result = check(model)
    except Exception:
        return False
    return bool(result)
