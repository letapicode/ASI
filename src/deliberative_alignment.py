import re
from typing import Iterable, List

from .normative_reasoner import NormativeReasoner


class DeliberativeAligner:
    """Minimal chain-of-thought policy checker for Plan.md L-2."""

    def __init__(self, policy_text: str, normative_rules: Iterable[str] | None = None) -> None:
        raw_rules = [r.strip().lower() for r in policy_text.splitlines() if r.strip()]
        self.rules: List[str] = []
        for r in raw_rules:
            if r.startswith("no "):
                r = r[3:]
            self.rules.append(r)
        self.normative = NormativeReasoner(normative_rules)

    def check(self, steps: Iterable[str]) -> bool:
        """Return ``True`` if every reasoning step complies with the policy."""
        ok, _ = self.normative.check(steps)
        if not ok:
            return False
        for step in steps:
            lower = step.lower()
            for rule in self.rules:
                if rule and re.search(re.escape(rule), lower):
                    return False
        return True

    def check_report(self, steps: Iterable[str]) -> tuple[bool, list[str]]:
        """Return ``(passed, normative_flagged)`` for ``steps``."""
        ok, flagged = self.normative.check(steps)
        if not ok:
            return False, flagged
        for step in steps:
            lower = step.lower()
            for rule in self.rules:
                if rule and re.search(re.escape(rule), lower):
                    return False, []
        return True, []

    def analyze(self, text: str) -> bool:
        """Split ``text`` into lines and check them sequentially."""
        steps = [s.strip() for s in text.splitlines() if s.strip()]
        return self.check(steps)
