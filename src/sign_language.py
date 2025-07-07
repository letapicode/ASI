import numpy as np

try:  # optional dependency
    import mediapipe as mp  # type: ignore
    import cv2  # type: ignore
    _HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover - optional
    mp = None
    cv2 = None
    _HAS_MEDIAPIPE = False


class SignLanguageRecognizer:
    """Extract simple sign embeddings and classify them."""

    def __init__(self, known_signs: dict[str, np.ndarray] | None = None) -> None:
        self.known_signs = {
            k: np.asarray(v, dtype=np.float32) for k, v in (known_signs or {}).items()
        }
        if _HAS_MEDIAPIPE:
            self._hands = mp.solutions.hands.Hands(static_image_mode=False)
        else:  # pragma: no cover - fallback
            self._hands = None

    # ------------------------------------------------------------------
    def encode(self, video: str | np.ndarray) -> np.ndarray:
        """Return a landmark embedding for ``video``."""
        if isinstance(video, np.ndarray):
            return np.asarray(video, dtype=np.float32)
        if not _HAS_MEDIAPIPE or self._hands is None:
            return np.zeros(63 * 2, dtype=np.float32)
        cap = cv2.VideoCapture(str(video))
        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._hands.process(frame)
            vec: list[float] = []
            if res.left_hand_landmarks:
                for lm in res.left_hand_landmarks.landmark:
                    vec.extend([lm.x, lm.y, lm.z])
            if res.right_hand_landmarks:
                for lm in res.right_hand_landmarks.landmark:
                    vec.extend([lm.x, lm.y, lm.z])
            if vec:
                frames.append(np.asarray(vec, dtype=np.float32))
        cap.release()
        if not frames:
            return np.zeros(63 * 2, dtype=np.float32)
        return np.stack(frames).mean(axis=0)

    # ------------------------------------------------------------------
    def recognize(self, video_or_emb: str | np.ndarray) -> str:
        """Return the closest known sign label or ``""``."""
        emb = self.encode(video_or_emb) if not isinstance(video_or_emb, np.ndarray) else np.asarray(video_or_emb, dtype=np.float32)
        best = ""
        best_score = -float("inf")
        for name, ref in self.known_signs.items():
            score = float(np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-8))
            if score > best_score:
                best_score = score
                best = name
        return best


__all__ = ["SignLanguageRecognizer"]
