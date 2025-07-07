from __future__ import annotations

import random
from typing import Tuple
import numpy as np


class PrivacyGuard:
    """Inject random noise and track a privacy budget."""

    def __init__(self, budget: float, noise_scale: float = 0.1) -> None:
        self.budget = float(budget)
        self.noise_scale = float(noise_scale)
        self._consumed = 0.0

    # --------------------------------------------------
    def _noisy_text(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        p = min(self.noise_scale, 0.5)
        kept = [w for w in words if random.random() > p]
        if not kept:
            kept = [words[0]]
        return " ".join(kept)

    def _noisy_array(self, arr: np.ndarray) -> np.ndarray:
        noise = np.random.normal(scale=self.noise_scale, size=arr.shape)
        out = arr.astype(float) + noise
        return out.astype(arr.dtype)

    # --------------------------------------------------
    def inject(
        self,
        text: str,
        image: np.ndarray,
        audio: np.ndarray,
        epsilon: float = 0.1,
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """Return noisy versions of ``text``/``image``/``audio``."""
        if self.remaining_budget() <= 0.0:
            return text, image, audio
        txt = self._noisy_text(text)
        img = self._noisy_array(np.asarray(image))
        aud = self._noisy_array(np.asarray(audio))
        self._consumed += epsilon
        return txt, img, aud

    # --------------------------------------------------
    def remaining_budget(self) -> float:
        return max(self.budget - self._consumed, 0.0)


__all__ = ["PrivacyGuard"]
