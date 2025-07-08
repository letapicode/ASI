from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, Sequence, Optional, Dict
import numpy as np

from .cognitive_load_monitor import CognitiveLoadMonitor
from .emotion_detector import detect_emotion


@dataclass
class CognitiveFeedbackLoop:
    """Combine physiological signals and emotion cues to adjust prompts."""

    load_monitor: CognitiveLoadMonitor
    adjust_fn: Callable[[str, Dict[str, Any]], str] | None = None
    history: list[Dict[str, Any]] = field(default_factory=list)

    # --------------------------------------------------------------
    def process(
        self,
        text: str,
        *,
        eeg: Optional[Sequence[float]] = None,
        gaze: Optional[Sequence[float]] = None,
    ) -> str:
        """Record signals and return the adjusted text."""
        info: Dict[str, Any] = {
            "emotion": detect_emotion(text),
            "load": self.load_monitor.cognitive_load(),
        }
        if eeg is not None:
            info["eeg"] = float(np.mean(eeg)) if len(eeg) else 0.0
        if gaze is not None:
            info["gaze"] = float(np.mean(gaze)) if len(gaze) else 0.0
        self.history.append(info)
        if self.adjust_fn is not None:
            return self.adjust_fn(text, info)
        return text


__all__ = ["CognitiveFeedbackLoop"]
