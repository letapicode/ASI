from __future__ import annotations
from typing import Dict


class FairnessEvaluator:
    """Compute simple fairness metrics."""

    def demographic_parity(self, label_counts: Dict[str, Dict[str, int]], positive_label: str = "1") -> float:
        rates = []
        for group, counts in label_counts.items():
            total = sum(counts.values())
            if total == 0:
                continue
            rates.append(counts.get(positive_label, 0) / total)
        if not rates:
            return 0.0
        return max(rates) - min(rates)

    def equal_opportunity(self, stats: Dict[str, Dict[str, int]]) -> float:
        tpr = []
        for group, counts in stats.items():
            pos = counts.get("tp", 0) + counts.get("fn", 0)
            if pos == 0:
                continue
            tpr.append(counts.get("tp", 0) / pos)
        if not tpr:
            return 0.0
        return max(tpr) - min(tpr)

    def evaluate(self, stats: Dict[str, Dict[str, int]], positive_label: str = "1") -> Dict[str, float]:
        return {
            "demographic_parity": self.demographic_parity(stats, positive_label),
            "equal_opportunity": self.equal_opportunity(stats),
        }

    def evaluate_multimodal(
        self, stats: Dict[str, Dict[str, Dict[str, int]]], positive_label: str = "1"
    ) -> Dict[str, Dict[str, float]]:
        """Return metrics for each modality in ``stats``.

        ``stats`` maps modality → group → label counts.
        """
        results: Dict[str, Dict[str, float]] = {}
        for modality, groups in stats.items():
            results[modality] = self.evaluate(groups, positive_label)
        return results

__all__ = ["FairnessEvaluator"]
