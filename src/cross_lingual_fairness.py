from __future__ import annotations

from typing import Dict

from .fairness_evaluator import FairnessEvaluator
try:  # pragma: no cover - optional dependency
    from .data_ingest import CrossLingualTranslator
except Exception:  # pragma: no cover - missing torch
    class CrossLingualTranslator:  # type: ignore
        def __init__(self, languages):
            self.languages = list(languages)

        def translate(self, text: str, lang: str) -> str:
            if lang not in self.languages:
                raise ValueError("unsupported language")
            return text

        def translate_all(self, text: str):
            return {l: text for l in self.languages}


class CrossLingualFairnessEvaluator:
    """FairnessEvaluator that normalizes groups across languages."""

    def __init__(
        self,
        translator: CrossLingualTranslator | None = None,
        target_lang: str = "en",
    ) -> None:
        self.translator = translator
        self.target_lang = target_lang
        self.base = FairnessEvaluator()

    def _translate(self, text: str) -> str:
        if self.translator is None:
            return text
        return self.translator.translate(text, self.target_lang)

    def _normalize(self, stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        norm: Dict[str, Dict[str, int]] = {}
        for group, counts in stats.items():
            g = self._translate(group)
            out = norm.setdefault(g, {})
            for label, cnt in counts.items():
                if label in {"tp", "fp", "fn", "tn"}:
                    l = label
                else:
                    l = self._translate(label)
                out[l] = out.get(l, 0) + cnt
        return norm

    def evaluate(self, stats: Dict[str, Dict[str, int]], positive_label: str = "1") -> Dict[str, float]:
        norm = self._normalize(stats)
        if positive_label in {"tp", "fp", "fn", "tn"}:
            pos = positive_label
        else:
            pos = self._translate(positive_label)
        return self.base.evaluate(norm, positive_label=pos)


__all__ = ["CrossLingualFairnessEvaluator"]
