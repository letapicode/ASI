from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Tuple


class NormativeReasoner:
    """Validate reasoning steps against configurable ethics rules.

    Parameters
    ----------
    rules:
        Sequence of rules expressed as plain text or regular expressions.
    use_regex:
        Treat each rule as a raw regular expression instead of escaping it.
    fuzzy_threshold:
        When set, apply fuzzy matching using :class:`difflib.SequenceMatcher` and
        flag any step with a similarity ratio greater than or equal to this
        threshold.  Values are in ``[0.0, 1.0]``.
    """

    def __init__(
        self,
        rules: Iterable[str] | None = None,
        *,
        use_regex: bool = False,
        fuzzy_threshold: float | None = None,
    ) -> None:
        cleaned: List[tuple[str, re.Pattern]] = []
        for r in (rules or []):
            raw = r.strip()
            if raw.lower().startswith("no "):
                raw = raw[3:]
            if not raw:
                continue
            pattern_text = raw if use_regex else re.escape(raw)
            cleaned.append((raw, re.compile(pattern_text, re.IGNORECASE)))

        self.rules = cleaned
        self.use_regex = use_regex
        self.fuzzy_threshold = fuzzy_threshold

    def check(self, steps: Iterable[str]) -> Tuple[bool, List[str]]:
        """Return ``(passed, flagged)`` for ``steps``."""
        flagged: List[str] = []
        for step in steps:
            for raw, pattern in self.rules:
                if pattern.search(step):
                    flagged.append(step)
                    break
                if self.fuzzy_threshold is not None:
                    ratio = SequenceMatcher(None, raw.lower(), step.lower()).ratio()
                    if ratio >= self.fuzzy_threshold:
                        flagged.append(step)
                        break
        return (not flagged, flagged)

    def analyze(self, text: str) -> Tuple[bool, List[str]]:
        """Split ``text`` and check each line."""
        steps = [s.strip() for s in text.splitlines() if s.strip()]
        return self.check(steps)


__all__ = ["NormativeReasoner"]
