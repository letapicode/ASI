from __future__ import annotations

import argparse
import json
from typing import Sequence

from .reasoning_history import ReasoningHistoryLogger


def main(argv: Sequence[str] | None = None) -> None:
    """Print a report of reasoning history analysis."""
    parser = argparse.ArgumentParser(description="Summarise reasoning history")
    parser.add_argument("history", help="Path to history JSON file")
    args = parser.parse_args(argv)

    logger = ReasoningHistoryLogger.load(args.history)
    analysis = logger.analyze()

    lines = ["Reasoning step clusters:"]
    for step, count in sorted(analysis.clusters.items(), key=lambda x: -x[1]):
        lines.append(f"- {step}: {count}")
    if analysis.inconsistencies:
        lines.append("Inconsistencies:")
        for a, b in analysis.inconsistencies:
            lines.append(f"- {a} vs {b}")

    languages: set[str] = set()
    for _ts, entry in logger.get_history():
        if isinstance(entry, dict):
            languages.update(entry.get("translations", {}).keys())

    translations: dict[str, str] = {}
    if languages:
        from .cross_lingual_translator import CrossLingualTranslator
        from .reasoning_summary_translator import ReasoningSummaryTranslator

        translator = CrossLingualTranslator(sorted(languages))
        rst = ReasoningSummaryTranslator(translator)
        info = rst.summarize(logger)
        lines = info["summary"].split("\n")
        translations = info["translations"]

    if translations:
        lines.append("Translations:")
        for lang, txt in translations.items():
            lines.append(f"- [{lang}] {txt}")

    print("\n".join(lines))


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
