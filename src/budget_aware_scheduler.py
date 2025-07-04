"""Adjust training config based on remaining GPU budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compute_budget_tracker import ComputeBudgetTracker


@dataclass
class BudgetAwareScheduler:
    """Reduce batch size and learning rate as the budget shrinks."""

    tracker: ComputeBudgetTracker
    run_id: str = "default"
    threshold: float = 1.0

    def schedule_step(self, config: Any) -> None:
        """Modify ``config`` in-place if budget is below ``threshold``."""
        if self.tracker.remaining(self.run_id) < self.threshold:
            if hasattr(config, "batch_size"):
                config.batch_size = max(1, config.batch_size // 2)
            if hasattr(config, "lr"):
                config.lr *= 0.5


__all__ = ["BudgetAwareScheduler"]
