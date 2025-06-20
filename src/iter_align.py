import re
from typing import Iterable, List

class IterativeAligner:
    """Iterative self-alignment helper for Plan.md L-3."""

    def __init__(self, initial_rules: Iterable[str]):
        self.rules: List[str] = [r.strip().lower() for r in initial_rules if r.strip()]

    def critique(self, text: str) -> List[str]:
        """Return lines that violate current rules."""
        flagged: List[str] = []
        for line in text.splitlines():
            clean = line.strip().lower()
            if not clean:
                continue
            for rule in self.rules:
                term = rule[3:] if rule.startswith("no ") else rule
                root = term[:-3] if term.endswith("ing") else term
                if term and (term in clean or root in clean):
                    flagged.append(clean)
                    break
        return flagged

    def refine(self, flagged: Iterable[str]) -> None:
        """Add new rules derived from ``flagged`` lines."""
        for line in flagged:
            rule = line.strip()
            if rule and rule not in self.rules:
                self.rules.append(rule)

    def iterate(self, transcripts: Iterable[str], rounds: int = 3) -> List[str]:
        """Run critique/refine loops over ``transcripts``."""
        for _ in range(rounds):
            flagged: List[str] = []
            for text in transcripts:
                flagged.extend(self.critique(text))
            if not flagged:
                break
            self.refine(flagged)
        return self.rules
