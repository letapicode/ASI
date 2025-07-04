from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RiskScoreboard:
    """Compute a simple ethical risk metric."""

    metrics: Dict[str, float] = field(default_factory=dict)

    def update(
        self, license_violations: int, privacy_cost: float, alignment_score: float
    ) -> float:
        risk = license_violations * 10 + privacy_cost * 0.1 - alignment_score
        self.metrics = {
            "license_violations": float(license_violations),
            "privacy_cost": float(privacy_cost),
            "alignment_score": float(alignment_score),
            "risk_score": float(risk),
        }
        return risk

    def get_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)


__all__ = ["RiskScoreboard"]
