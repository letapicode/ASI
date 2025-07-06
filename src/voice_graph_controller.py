from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .nl_graph_editor import NLGraphEditor
from .data_ingest import CrossLingualTranslator, CrossLingualSpeechTranslator

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


__all__ = ["VoiceGraphController"]
