from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class BudgetRecord:
    epsilon: float = 0.0
    delta: float = 0.0


class PrivacyBudgetManager:
    """Track cumulative privacy loss across training runs."""

    def __init__(self, budget: float, delta: float = 1e-5, log_path: str | Path = "privacy_budget.json") -> None:
        self.budget = budget
        self.delta_budget = delta
        self.path = Path(log_path)
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.records: Dict[str, BudgetRecord] = {k: BudgetRecord(**v) for k, v in data.items()}
        else:
            self.records = {}

    # --------------------------------------------------------------
    def consume(self, run_id: str, epsilon: float, delta: float) -> None:
        rec = self.records.get(run_id, BudgetRecord())
        rec.epsilon += epsilon
        rec.delta += delta
        self.records[run_id] = rec
        self.path.write_text(json.dumps({k: asdict(v) for k, v in self.records.items()}, indent=2))

    # --------------------------------------------------------------
    def remaining(self, run_id: str) -> Tuple[float, float]:
        rec = self.records.get(run_id, BudgetRecord())
        return max(self.budget - rec.epsilon, 0.0), max(self.delta_budget - rec.delta, 0.0)


__all__ = ["PrivacyBudgetManager", "BudgetRecord"]
