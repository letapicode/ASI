from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Mapping


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
        Format string for replacement tags. ``"[{label}]"`` by default.
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
        p = Path(path)
        p.write_text(self.anonymize(p.read_text()))

    # --------------------------------------------------------------
    def summary(self) -> Dict[str, int]:
        return dict(self.counts)


__all__ = ["NERAnonymizer"]
