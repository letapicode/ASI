from __future__ import annotations

import random
from typing import Dict

from .cross_lingual_translator import CrossLingualTranslator


class AdaptiveTranslator:
    """Learn which language yields better translations using Q-values."""

    def __init__(self, translator: CrossLingualTranslator, epsilon: float = 0.1, alpha: float = 0.5) -> None:
        self.translator = translator
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.q: Dict[str, float] = {l: 0.0 for l in translator.languages}
        self.last_lang: str | None = None

    # ------------------------------------------------------------------
    def choose_language(self) -> str:
        """Return a language index using epsilon-greedy selection."""
        if random.random() < self.epsilon:
            return random.choice(self.translator.languages)
        return max(self.translator.languages, key=lambda l: self.q.get(l, 0.0))

    def translate(self, text: str, lang: str | None = None) -> str:
        """Translate ``text`` using ``lang`` or a learned preference."""
        if lang is None:
            lang = self.choose_language()
        self.last_lang = lang
        return self.translator._basic_translate(text, lang)

    def update(self, reward: float, lang: str | None = None) -> None:
        """Update Q-value for ``lang`` (defaults to last)."""
        l = lang or self.last_lang
        if l is None:
            return
        if l not in self.translator.languages:
            raise ValueError(f"unsupported language: {l}")
        current = self.q.get(l, 0.0)
        self.q[l] = current + self.alpha * (reward - current)


__all__ = ["AdaptiveTranslator"]
