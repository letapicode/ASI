from __future__ import annotations

from typing import Any, Dict, Mapping, TYPE_CHECKING

from .nl_graph_editor import NLGraphEditor
from .data_ingest import CrossLingualTranslator, CrossLingualSpeechTranslator
from .sign_language import SignLanguageRecognizer

if TYPE_CHECKING:  # pragma: no cover - for type hints
    import numpy as np
    import torch


class VoiceGraphController:
    """Convert speech to text and forward commands to ``NLGraphEditor``."""

    def __init__(self, editor: NLGraphEditor) -> None:
        self.editor = editor
        self.transcriber = CrossLingualSpeechTranslator(CrossLingualTranslator([]))

    def apply(
        self, audio: str | "np.ndarray" | "torch.Tensor"
    ) -> Dict[str, Any]:
        """Transcribe ``audio`` and apply the resulting command."""
        text = self.transcriber.transcribe(audio)
        if not text:
            raise ValueError("could not transcribe audio")
        return self.editor.apply(text)


class SignLanguageGraphController:
    """Interpret sign-language video and forward commands across languages."""

    def __init__(
        self,
        editor: NLGraphEditor,
        translator: CrossLingualTranslator | None = None,
        mapping: Mapping[str, str] | None = None,
    ) -> None:
        self.editor = editor
        self.recognizer = SignLanguageRecognizer()
        self.translator = translator or CrossLingualTranslator([])
        self.mapping = dict(mapping or {"hello": "add node hello", "thanks": "add node thanks"})

    def apply(self, video: "np.ndarray") -> Dict[str, Any]:
        """Recognize ``video`` and apply the translated command."""
        gesture = self.recognizer.recognize(video)
        if not gesture:
            raise ValueError("could not recognize sign language")
        base_cmd = self.mapping.get(gesture, gesture)
        attempts = [base_cmd]
        for lang in self.translator.languages:
            try:
                attempts.append(self.translator.translate(base_cmd, lang))
            except Exception:
                continue
        for text in attempts:
            try:
                return self.editor.apply(text)
            except Exception:
                continue
        raise ValueError("unrecognized command")


__all__ = ["VoiceGraphController", "SignLanguageGraphController"]
