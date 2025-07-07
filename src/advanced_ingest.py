from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import ClassVar, List, Tuple

try:
    import spacy  # type: ignore
    _HAS_SPACY = True
except Exception:  # pragma: no cover - optional
    spacy = None  # type: ignore
    _HAS_SPACY = False


class LLMIngestParser:
    """Parse text into subject-verb-object triples using a lightweight model."""

    _shared_nlp: ClassVar[object | None] = None

    def __init__(self, model: str | None = None) -> None:
        """Initialize the parser.

        Parameters
        ----------
        model:
            Name or path of the spaCy model to load. If ``None``, the
            ``LLM_PARSER_MODEL`` environment variable or ``en_core_web_sm`` is
            used. The loaded model is cached so subsequent instances reuse it.
        """

        if model is None:
            model = os.environ.get("LLM_PARSER_MODEL", "en_core_web_sm")

        if _HAS_SPACY and LLMIngestParser._shared_nlp is None:
            try:
                LLMIngestParser._shared_nlp = spacy.load(model)  # type: ignore
            except Exception:  # pragma: no cover - fallback
                LLMIngestParser._shared_nlp = None

        self.nlp = LLMIngestParser._shared_nlp

    def parse(self, text: str) -> List[Tuple[str, str, str]]:
        if self.nlp is not None:
            triples: List[Tuple[str, str, str]] = []
            doc = self.nlp(text)
            for sent in doc.sents:
                subj = None
                obj = None
                verb = None
                for tok in sent:
                    if tok.dep_ == "nsubj" and subj is None:
                        subj = tok.text
                    elif tok.dep_ in {"dobj", "pobj"} and obj is None:
                        obj = tok.text
                    elif tok.pos_ == "VERB" and verb is None:
                        verb = tok.lemma_
                if subj and verb and obj:
                    triples.append((subj, verb, obj))
            if triples:
                return triples

        # Heuristic fallback when no model is available
        triples = []
        for sent in re.split(r"[.!?]+", text):
            words = re.findall(r"\b\w+\b", sent)
            if len(words) < 3:
                continue
            subj, verb, *rest = words
            obj = " ".join(rest) if rest else ""
            triples.append((subj, verb, obj))
        return triples

    def parse_file(self, path: str | Path) -> List[Tuple[str, str, str]]:
        try:
            text = Path(path).read_text()
        except Exception:  # pragma: no cover - file read failure
            return []
        return self.parse(text)

    def parse_file_to_json(self, path: str | Path, cache: bool = True) -> Path:
        """Parse ``path`` and write triples to a ``.triples.json`` file."""

        out = Path(path).with_suffix(".triples.json")
        if cache and out.exists():
            return out
        triples = self.parse_file(path)
        out.write_text(json.dumps(triples))
        return out
