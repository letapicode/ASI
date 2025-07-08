"""Index a small reasoning history and evaluate cross-lingual search.

The demo uses a caching translator so repeated translations do not
incur additional overhead."""
from __future__ import annotations

import argparse
from typing import Iterable, Dict

from asi.cross_lingual_memory import CrossLingualMemory


class PrefixTranslator:
    """Simple placeholder translator that handles prefixed text."""

    def __init__(self, languages: Iterable[str]) -> None:
        self.languages = list(languages)

    def _strip(self, text: str) -> str:
        if text.startswith("[") and "]" in text:
            return text.split("]", 1)[1].lstrip()
        return text

    def translate(self, text: str, lang: str) -> str:
        return f"[{lang}] {self._strip(text)}"

    def translate_all(self, text: str) -> Dict[str, str]:
        base = self._strip(text)
        return {l: f"[{l}] {base}" for l in self.languages}


class CachingTranslator(PrefixTranslator):
    """Translate once and reuse results for identical input."""

    def __init__(self, languages: Iterable[str]) -> None:
        super().__init__(languages)
        self._cache: Dict[str, Dict[str, str]] = {}

    def translate(self, text: str, lang: str) -> str:  # type: ignore[override]
        if text not in self._cache:
            self._cache[text] = super().translate_all(text)
        return self._cache[text][lang]

    def translate_all(self, text: str) -> Dict[str, str]:  # type: ignore[override]
        if text not in self._cache:
            self._cache[text] = super().translate_all(text)
        return self._cache[text]


def build_memory(dim: int = 16) -> tuple[CrossLingualMemory, list[str], CachingTranslator]:
    history = [
        "Analyze training data quality",
        "Plan memory optimization experiments",
        "Evaluate cross-lingual retrieval",
    ]
    translator = CachingTranslator(["es", "fr"])
    mem = CrossLingualMemory(
        dim=dim,
        compressed_dim=dim // 2,
        capacity=100,
        translator=translator,
        encryption_key=b"0" * 32,
    )
    mem.add_texts(history, metadata=list(range(len(history))))
    return mem, history, translator


def evaluate(mem: CrossLingualMemory, history: list[str], translator: CachingTranslator) -> float:
    correct = 0
    total = 0
    for idx, text in enumerate(history):
        for lang in translator.languages:
            query = translator.translate(text, lang)
            _vecs, meta = mem.search(query, k=1)
            if meta and meta[0] == idx:
                correct += 1
            total += 1
    return correct / total if total else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-lingual reasoning demo")
    parser.add_argument("--dim", type=int, default=16)
    args = parser.parse_args(argv)

    mem, history, translator = build_memory(args.dim)
    acc = evaluate(mem, history, translator)
    print(f"cross_lingual_accuracy: {acc:.2f}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
