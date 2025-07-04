from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


class DatasetAnonymizer:
    """Scrub common PII from text, images and audio."""

    EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+")
    PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

    def __init__(self) -> None:
        self.counts = {"text": 0, "image": 0, "audio": 0}

    # --------------------------------------------------------------
    def scrub_text(self, text: str) -> str:
        new = self.EMAIL_RE.sub("[EMAIL]", text)
        new = self.PHONE_RE.sub("[PHONE]", new)
        if new != text:
            self.counts["text"] += 1
        return new

    # --------------------------------------------------------------
    def scrub_image(self, image: Image.Image) -> Image.Image:
        arr = np.array(image)
        arr[:] = 0
        self.counts["image"] += 1
        return Image.fromarray(arr)

    # --------------------------------------------------------------
    def scrub_audio(self, audio: np.ndarray) -> np.ndarray:
        out = np.zeros_like(audio)
        self.counts["audio"] += 1
        return out

    # --------------------------------------------------------------
    def scrub_text_file(self, path: str | Path) -> None:
        p = Path(path)
        p.write_text(self.scrub_text(p.read_text()))

    # --------------------------------------------------------------
    def scrub_image_file(self, path: str | Path) -> None:
        img = Image.open(path)
        self.scrub_image(img).save(path)

    # --------------------------------------------------------------
    def scrub_audio_file(self, path: str | Path) -> None:
        with wave.open(str(path), "rb") as f:
            params = f.getparams()
            frames = f.readframes(f.getnframes())
        arr = np.frombuffer(frames, dtype=np.int16)
        clean = self.scrub_audio(arr)
        with wave.open(str(path), "wb") as f:
            f.setparams(params)
            f.writeframes(clean.tobytes())

    # --------------------------------------------------------------
    def summary(self) -> Dict[str, int]:
        return dict(self.counts)


__all__ = ["DatasetAnonymizer"]
