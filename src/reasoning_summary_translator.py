from __future__ import annotations

from typing import Dict

from .data_ingest import CrossLingualTranslator
from .reasoning_history import ReasoningHistoryLogger


class ReasoningSummaryTranslator:
    """Summarize reasoning histories and translate the summary."""

    def __init__(self, translator: CrossLingualTranslator) -> None:
        self.translator = translator

    # --------------------------------------------------------------
    def summarize(self, logger: ReasoningHistoryLogger) -> Dict[str, object]:
        """Return cluster summary and translations."""
        results = logger.analyze()
        lines = ["Reasoning step clusters:"]
        for step, count in sorted(
            results.clusters.items(), key=lambda x: (-x[1], x[0])
        ):
            lines.append(f"- {step}: {count}")
        if results.inconsistencies:
            lines.append("Inconsistencies:")
            for a, b in results.inconsistencies:
                lines.append(f"- {a} vs {b}")
        summary = "\n".join(lines)
        return {
            "summary": summary,
            "translations": self.translator.translate_all(summary),
        }


__all__ = ["ReasoningSummaryTranslator"]
