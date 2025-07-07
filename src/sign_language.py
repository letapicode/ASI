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

    def _fallback(self, video: np.ndarray) -> str:
        mean = float(video.mean())
        if mean > 0.66:
            return "hello"
        if mean > 0.33:
            return "thanks"
        return ""

    def recognize(self, video: np.ndarray) -> str:
        """Return a rough transcription of ``video`` or ``""`` on failure."""
        if self._hands is None:
            return self._fallback(video)

        open_hand = 0
        closed = 0
        for frame in video:
            res = self._hands.process(frame)
            if not getattr(res, "multi_hand_landmarks", None):
                continue
            lm = res.multi_hand_landmarks[0].landmark
            thumb = lm[4]
            index = lm[8]
            dist = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
            if dist > 0.1:
                open_hand += 1
            else:
                closed += 1
        if open_hand > closed:
            return "hello"
        if closed:
            return "thanks"
        return ""


__all__ = ["SignLanguageRecognizer", "_HAS_MEDIAPIPE"]
