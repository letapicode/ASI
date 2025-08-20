from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, TYPE_CHECKING

from .graph_of_thought import GraphOfThought
from .cross_lingual_translator import CrossLingualSpeechTranslator

if TYPE_CHECKING:  # pragma: no cover - for type hints
    import numpy as np
    import torch


class CrossLingualVoiceChat:
    """Simple voice chat agent using speech-to-text and a reasoning graph."""

    def __init__(
        self,
        translator: CrossLingualSpeechTranslator,
        graph: GraphOfThought | None = None,
        history: int = 6,
    ) -> None:
        self.translator = translator
        self.graph = graph or GraphOfThought()
        self.history: Deque[tuple[str, int]] = deque(maxlen=history)
        try:  # pragma: no cover - optional dependency
            import pyttsx3  # type: ignore

            self._tts = pyttsx3.init()
        except Exception:  # pragma: no cover - missing package
            self._tts = None

    # --------------------------------------------------------------
    def _speak(self, text: str) -> bytes | None:
        if self._tts is None:
            return None
        import tempfile
        from pathlib import Path

        path = tempfile.mktemp(suffix=".wav")
        self._tts.save_to_file(text, path)
        self._tts.runAndWait()
        data = Path(path).read_bytes()
        Path(path).unlink(missing_ok=True)
        return data

    # --------------------------------------------------------------
    def chat(self, audio: str | "np.ndarray" | "torch.Tensor") -> Dict[str, Any]:
        """Process ``audio`` and return a response with optional speech audio."""
        text = self.translator.transcribe(audio)
        if not text:
            raise ValueError("could not transcribe audio")
        translations = self.translator.translator.translate_all(text)
        meta = {"translations": translations}
        user = self.graph.add_step(text, metadata=meta)
        if self.history:
            try:
                self.graph.connect(self.history[-1][1], user)
            except Exception:
                pass
        self.history.append(("user", user))
        reply = f"You said: {text}"
        agent = self.graph.add_step(reply)
        self.graph.connect(user, agent)
        self.history.append(("agent", agent))
        audio_data = self._speak(reply)
        result = {"text": reply}
        if audio_data is not None:
            import base64

            result["audio"] = base64.b64encode(audio_data).decode()
        return result


__all__ = ["CrossLingualVoiceChat"]
