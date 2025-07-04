import torch
from torch.nn import functional as F

class SemanticDriftDetector:
    """Compare successive model outputs and track drift."""

    def __init__(self) -> None:
        self.prev_probs: torch.Tensor | None = None
        self.history: list[float] = []

    def update(self, logits: torch.Tensor) -> float:
        """Update with ``logits`` and return KL divergence from previous step."""
        probs = F.softmax(logits.detach(), dim=-1)
        if self.prev_probs is None:
            self.prev_probs = probs
            self.history.append(0.0)
            return 0.0
        p = self.prev_probs
        q = probs
        kl = F.kl_div(q.log(), p, reduction="batchmean")
        self.prev_probs = probs
        val = float(kl.item())
        self.history.append(val)
        return val

    def last_drift(self) -> float:
        return self.history[-1] if self.history else 0.0

__all__ = ["SemanticDriftDetector"]
