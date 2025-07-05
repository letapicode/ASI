from __future__ import annotations

import random
from typing import Callable, List, Tuple, Optional

from .user_preferences import UserPreferences

class PromptOptimizer:
    """Simple prompt optimizer using random mutations and acceptance by score.

    If ``user_preferences`` is supplied, the scoring function is augmented with
    the dot product between the user's preference vector and the prompt
    embedding.
    """

    def __init__(
        self,
        scorer: Callable[[str], float],
        base_prompt: str,
        lr: float = 0.1,
        user_preferences: Optional[UserPreferences] = None,
        user_id: Optional[str] = None,
    ) -> None:
        self.scorer = scorer
        self.prompt = base_prompt
        self.lr = lr
        self.history: List[Tuple[str, float]] = [(base_prompt, self._score(base_prompt))]
        self.user_preferences = user_preferences
        self.user_id = user_id

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

    # ------------------------------------------------------------
    def _score(self, prompt: str) -> float:
        score = self.scorer(prompt)
        if self.user_preferences and self.user_id is not None:
            pref = self.user_preferences.get_vector(self.user_id)
            emb = self.user_preferences.embed_text(prompt)
            score += float(pref @ emb)
        return score

    def step(self) -> str:
        """Mutate the current prompt and keep it if score improves."""
        candidate = self._mutate(self.prompt)
        new_score = self._score(candidate)
        old_score = self._score(self.prompt)
        if new_score >= old_score or random.random() < self.lr:
            self.prompt = candidate
            self.history.append((candidate, new_score))
        return self.prompt

    def optimize(self, steps: int = 10) -> str:
        for _ in range(steps):
            self.step()
        return self.prompt

__all__ = ["PromptOptimizer"]
