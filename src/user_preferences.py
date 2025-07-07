from __future__ import annotations

from typing import Dict, Tuple, List
import numpy as np


class UserPreferences:
    """Maintain per-user preference vectors and feedback stats."""

    def __init__(self, dim: int = 16, history_size: int = 10) -> None:
        self.dim = dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.stats: Dict[str, Dict[str, int]] = {}
        self.languages: Dict[str, str] = {}
        self.emotions: Dict[str, str] = {}
        self.emotion_history: Dict[str, list[str]] = {}
        self.history_size = history_size

    # --------------------------------------------------------------
    def embed_text(self, text: str) -> np.ndarray:
        """Hash words into a fixed-size embedding."""
        vec = np.zeros(self.dim, dtype=np.float32)
        for w in text.split():
            vec[hash(w) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # --------------------------------------------------------------
    def get_vector(self, user_id: str) -> np.ndarray:
        return self.vectors.get(user_id, np.zeros(self.dim, dtype=np.float32))

    # --------------------------------------------------------------
    def get_stats(self, user_id: str) -> Tuple[int, int]:
        st = self.stats.get(user_id, {"pos": 0, "neg": 0})
        return st["pos"], st["neg"]

    # --------------------------------------------------------------
    def set_language(self, user_id: str, language: str) -> None:
        """Store the preferred ``language`` for ``user_id``."""
        self.languages[user_id] = language

    def get_language(self, user_id: str) -> str | None:
        return self.languages.get(user_id)

    # --------------------------------------------------------------
    def set_emotion(self, user_id: str, emotion: str) -> None:
        """Store the detected ``emotion`` for ``user_id`` and update history."""
        self.emotions[user_id] = emotion
        hist = self.emotion_history.setdefault(user_id, [])
        hist.append(emotion)
        if len(hist) > self.history_size:
            del hist[0]

    def get_emotion(self, user_id: str) -> str | None:
        return self.emotions.get(user_id)

    def get_emotion_history(self, user_id: str) -> List[str]:
        return list(self.emotion_history.get(user_id, []))

    # --------------------------------------------------------------
    def update(self, user_id: str, vector: np.ndarray, feedback: float = 1.0) -> None:
        arr = np.asarray(vector, dtype=np.float32).reshape(self.dim)
        sign = 1.0 if feedback >= 0 else -1.0
        st = self.stats.setdefault(user_id, {"pos": 0, "neg": 0})
        if sign > 0:
            st["pos"] += 1
        else:
            st["neg"] += 1
        count = st["pos"] + st["neg"]
        prev = self.vectors.get(user_id, np.zeros(self.dim, dtype=np.float32))
        self.vectors[user_id] = (prev * (count - 1) + sign * arr) / float(count)

    # --------------------------------------------------------------
    def update_user_text(self, user_id: str, text: str, feedback: float = 1.0) -> None:
        self.update(user_id, self.embed_text(text), feedback)


__all__ = ["UserPreferences"]
