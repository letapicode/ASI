from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

@dataclass
class TrainingAnomalyDetector:
    """Detect sudden spikes in loss during training."""

    window: int = 10
    threshold: float = 2.0
    losses: deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def record(self, loss: float) -> bool:
        """Record ``loss`` and return ``True`` if it is anomalous."""
        self.losses.append(loss)
        if len(self.losses) < 2:
            return False
        avg = sum(self.losses) / len(self.losses)
        return loss > avg * self.threshold

__all__ = ["TrainingAnomalyDetector"]
