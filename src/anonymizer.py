from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
from PIL import Image

from .anonymizer_utils import rewrite_text_file


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
        rewrite_text_file(path, self.scrub_text)

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


class NERAnonymizer:
    """Replace named entities with tags using spaCy or regex fallbacks.

    Parameters
    ----------
    model:
        spaCy model name to load.
    extra_patterns:
        Optional mapping of entity labels to regex patterns or name lists used
        when spaCy is unavailable.
    tag_format:
        Format string for replacement tags. "[{label}]" by default.
    """

    PERSON_RE = re.compile(r"\b(Alice|Bob|Charlie)\b")
    ORG_RE = re.compile(r"\b(OpenAI|Google|NASA)\b")

    def __init__(
        self,
        model: str = "en_core_web_sm",
        *,
        extra_patterns: Mapping[str, Iterable[str] | re.Pattern[str]] | None = None,
        tag_format: str = "[{label}]",
    ) -> None:
        try:
            import spacy  # type: ignore

            try:
                self._nlp = spacy.load(model)
            except Exception:  # pragma: no cover - download at runtime
                try:
                    from spacy.cli import download  # type: ignore

                    download(model)
                    self._nlp = spacy.load(model)
                except Exception:
                    self._nlp = None
        except Exception:  # pragma: no cover - spaCy unavailable
            self._nlp = None
        self.counts: Dict[str, int] = {}
        self.tag_format = tag_format
        self.regex_patterns: Dict[str, re.Pattern[str]] = {
            "PERSON": self.PERSON_RE,
            "ORG": self.ORG_RE,
        }
        if extra_patterns:
            for label, pat in extra_patterns.items():
                if isinstance(pat, re.Pattern):
                    self.regex_patterns[label] = pat
                else:
                    self.regex_patterns[label] = re.compile(
                        r"\b(" + "|".join(re.escape(p) for p in pat) + r")\b"
                    )

    # --------------------------------------------------------------
    def anonymize(self, text: str) -> str:
        """Return ``text`` with entities replaced by ``[TYPE]`` tags."""
        out = text
        if self._nlp is not None:
            doc = self._nlp(text)
            for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
                label = ent.label_.upper()
                tag = self.tag_format.format(label=label)
                out = out[: ent.start_char] + tag + out[ent.end_char :]
                self.counts[label] = self.counts.get(label, 0) + 1
            return out

        for label, pattern in self.regex_patterns.items():
            def repl(match: re.Match[str], label: str = label) -> str:
                self.counts[label] = self.counts.get(label, 0) + 1
                return self.tag_format.format(label=label)

            out = pattern.sub(repl, out)
        return out

    # --------------------------------------------------------------
    def scrub_text_file(self, path: str | Path) -> None:
        rewrite_text_file(path, self.anonymize)

    # --------------------------------------------------------------
    def summary(self) -> Dict[str, int]:
        return dict(self.counts)


__all__ = ["DatasetAnonymizer", "NERAnonymizer"]
