import re
from typing import Iterable, List


class DeliberativeAligner:
    """Minimal chain-of-thought policy checker for Plan.md L-2."""

    def __init__(self, policy_text: str) -> None:
        raw_rules = [r.strip().lower() for r in policy_text.splitlines() if r.strip()]
        self.rules: List[str] = []
        for r in raw_rules:
            if r.startswith("no "):
                r = r[3:]
            self.rules.append(r)

    def check(self, steps: Iterable[str]) -> bool:
        """Return ``True`` if every reasoning step complies with the policy."""
        for step in steps:
            lower = step.lower()
            for rule in self.rules:
                if rule and re.search(re.escape(rule), lower):
                    return False
        return True

    def analyze(self, text: str) -> bool:
        """Split ``text`` into lines and check them sequentially."""
        steps = [s.strip() for s in text.splitlines() if s.strip()]
        return self.check(steps)
