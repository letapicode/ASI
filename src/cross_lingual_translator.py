from __future__ import annotations

"""Lightweight cross-lingual translation helpers."""

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import types
try:  # optional torch dependency
    import torch
except Exception:  # pragma: no cover - allow running without torch
    class _DummyTorch(types.SimpleNamespace):
        Tensor = type("Tensor", (), {})
    torch = _DummyTorch()


class CrossLingualTranslator:
    """Translate text to multiple languages using simple placeholders."""

    def __init__(
        self,
        languages: Iterable[str],
        adaptive: "AdaptiveTranslator | None" = None,
    ) -> None:
        self.languages = list(languages)
        self.adaptive = adaptive

    def _basic_translate(self, text: str, lang: str) -> str:
        if lang not in self.languages:
            raise ValueError(f"unsupported language: {lang}")
        return f"[{lang}] {text}"

    def translate(self, text: str, lang: str | None = None) -> str:
        if lang is None and self.adaptive is not None:
            return self.adaptive.translate(text)
        if lang is None:
            raise ValueError("language must be specified")
        return self._basic_translate(text, lang)

    def translate_all(self, text: str) -> Dict[str, str]:
        return {l: self._basic_translate(text, l) for l in self.languages}


class CrossLingualSpeechTranslator:
    """Offline speech-to-text wrapper using ``speech_recognition``."""

    def __init__(self, translator: CrossLingualTranslator) -> None:
        self.translator = translator
        try:
            import speech_recognition as sr  # type: ignore

            self._sr = sr.Recognizer()
            self._AudioFile = sr.AudioFile
            self._recognize = self._sr.recognize_sphinx
        except Exception:  # pragma: no cover - optional dependency
            self._sr = None
            self._AudioFile = None
            self._recognize = None

    def transcribe(self, audio: str | np.ndarray | torch.Tensor) -> str:
        """Return the transcript of ``audio`` or ``""`` on failure."""
        if self._sr is None or self._AudioFile is None or self._recognize is None:
            return ""

        import tempfile
        import wave

        path: str
        cleanup = False
        if isinstance(audio, str):
            path = audio
        else:
            arr = (
                audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else audio
            )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f.name, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    arr16 = np.asarray(arr, dtype=np.int16)
                    w.writeframes(arr16.tobytes())
                path = f.name
                cleanup = True

        try:
            with self._AudioFile(path) as source:
                data = self._sr.record(source)
            return self._recognize(data)
        except Exception:
            return ""
        finally:
            if cleanup:
                Path(path).unlink(missing_ok=True)

    def translate_all(self, audio: str | np.ndarray | torch.Tensor) -> Dict[str, str]:
        """Return translations for the transcript of ``audio``."""
        txt = self.transcribe(audio)
        return self.translator.translate_all(txt) if txt else {}


__all__ = [
    "CrossLingualTranslator",
    "CrossLingualSpeechTranslator",
]
