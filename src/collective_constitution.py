from collections import Counter
from typing import Iterable, List, Tuple

class CollectiveConstitution:
    """Aggregate crowd-sourced principles and label data."""

    def __init__(self, min_agreement: int = 2) -> None:
        self.min_agreement = min_agreement

    def derive_rules(self, principles: Iterable[str]) -> List[str]:
        """Return rules appearing at least ``min_agreement`` times."""
        counts = Counter(p.strip().lower() for p in principles)
        return [rule for rule, c in counts.items() if c >= self.min_agreement]

    def label_responses(self, responses: Iterable[str], rules: Iterable[str]) -> List[Tuple[str, bool]]:
        """Label responses as safe if they do not violate any rule."""
        compiled = [r.lower() for r in rules]
        labelled = []
        for resp in responses:
            text = resp.lower()
            safe = not any(rule in text for rule in compiled)
            labelled.append((resp, safe))
        return labelled

