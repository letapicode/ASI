from __future__ import annotations

from typing import Callable, List


class AdversarialRobustnessSuite:
    """Generate simple adversarial prompts for evaluation."""

    def __init__(self, model: Callable[[str], float]) -> None:
        self.model = model

    def generate(self, prompt: str, candidates: List[str]) -> str:
        """Return the most adversarial variant from ``candidates``."""
        base = self.model(prompt)
        worst = prompt
        worst_score = base
        for c in candidates:
            s = self.model(c)
            if s < worst_score:
                worst_score = s
                worst = c
        return worst

__all__ = ["AdversarialRobustnessSuite"]
