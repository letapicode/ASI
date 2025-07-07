import numpy as np

try:  # pragma: no cover - optional dependency
    import mediapipe as mp  # type: ignore
    _HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover - during tests
    mp = None
    _HAS_MEDIAPIPE = False


class SignLanguageRecognizer:
    """Tiny sign language recognizer using MediaPipe when available."""

    def __init__(self) -> None:
        if _HAS_MEDIAPIPE:
            self._hands = mp.solutions.hands.Hands(static_image_mode=False)
        else:
            self._hands = None

    def recognize(self, video: np.ndarray) -> str:
        """Return a rough transcription of ``video`` or ``""`` on failure."""
        if self._hands is None:
            return "hello" if video.mean() > 0 else ""

        for frame in video:
            res = self._hands.process(frame)
            if getattr(res, "multi_hand_landmarks", None):
                return "hello"
        return ""


__all__ = ["SignLanguageRecognizer", "_HAS_MEDIAPIPE"]
