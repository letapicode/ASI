from __future__ import annotations
from typing import Dict

import numpy as np


class FairnessEvaluator:
    """Compute simple fairness metrics."""

    def demographic_parity(self, label_counts: Dict[str, Dict[str, int]], positive_label: str = "1") -> float:
        rates = [
            counts.get(positive_label, 0) / total
            for counts in label_counts.values()
            if (total := sum(counts.values())) > 0
        ]
        if not rates:
            return 0.0
        arr = np.asarray(rates, dtype=float)
        return float(arr.max() - arr.min())

    def equal_opportunity(self, stats: Dict[str, Dict[str, int]]) -> float:
        tpr = [
            counts.get("tp", 0) / pos
            for counts in stats.values()
            if (pos := counts.get("tp", 0) + counts.get("fn", 0)) > 0
        ]
        if not tpr:
            return 0.0
        arr = np.asarray(tpr, dtype=float)
        return float(arr.max() - arr.min())

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
