from __future__ import annotations

from typing import Any

from .reasoning_history import ReasoningHistoryLogger


class RLDecisionNarrator:
    """Record RL action choices with short explanations."""

    def __init__(self, logger: ReasoningHistoryLogger | None = None) -> None:
        self.logger = logger or ReasoningHistoryLogger()

    def record_decision(self, state: Any, action: Any) -> None:
        """Log the selected action for ``state``."""
        summary = f"state={state!r} -> action={action}"
        self.logger.log(summary)

    def get_history(self) -> list[tuple[str, Any]]:
        return self.logger.get_history()


__all__ = ["RLDecisionNarrator"]
