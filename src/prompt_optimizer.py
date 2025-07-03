from __future__ import annotations

import random
from typing import Callable, List, Tuple

class PromptOptimizer:
    """Simple prompt optimizer using random mutations and acceptance by score."""

    def __init__(self, scorer: Callable[[str], float], base_prompt: str, lr: float = 0.1) -> None:
        self.scorer = scorer
        self.prompt = base_prompt
        self.lr = lr
        self.history: List[Tuple[str, float]] = [(base_prompt, scorer(base_prompt))]

    # ------------------------------------------------------------
    def _mutate(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        i = random.randrange(len(words))
        if random.random() < 0.5 and len(words) > 1:
            del words[i]
        else:
            words.insert(i, words[i])
        return " ".join(words)

    def step(self) -> str:
        """Mutate the current prompt and keep it if score improves."""
        candidate = self._mutate(self.prompt)
        new_score = self.scorer(candidate)
        old_score = self.scorer(self.prompt)
        if new_score >= old_score or random.random() < self.lr:
            self.prompt = candidate
            self.history.append((candidate, new_score))
        return self.prompt

    def optimize(self, steps: int = 10) -> str:
        for _ in range(steps):
            self.step()
        return self.prompt

__all__ = ["PromptOptimizer"]
