from __future__ import annotations

from typing import Dict, Any

try:
    from .cross_lingual_fairness import CrossLingualFairnessEvaluator
except Exception:  # pragma: no cover - during tests
    from cross_lingual_fairness import CrossLingualFairnessEvaluator  # type: ignore

try:
    from .dataset_weight_agent import DatasetWeightAgent
except Exception:  # pragma: no cover - during tests
    from dataset_weight_agent import DatasetWeightAgent  # type: ignore

try:
    from .data_ingest import ActiveDataSelector
except Exception:  # pragma: no cover - during tests
    from data_ingest import ActiveDataSelector  # type: ignore

try:
    from .adaptive_curriculum import SampleWeightRL
except Exception:  # pragma: no cover - during tests
    from adaptive_curriculum import SampleWeightRL  # type: ignore

try:
    from .dataset_lineage_manager import DatasetLineageManager
except Exception:  # pragma: no cover - during tests
    from dataset_lineage_manager import DatasetLineageManager  # type: ignore


class FairnessFeedback:
    """Adjust dataset selection based on cross-lingual fairness metrics."""

    def __init__(
        self,
        selector: ActiveDataSelector | None = None,
        weight_agent: DatasetWeightAgent | None = None,
        gap_threshold: float = 0.1,
        lineage: DatasetLineageManager | None = None,
        rl_lr: float = 0.1,
        positive_label: str = "tp",
        min_threshold: float = 0.1,
        max_threshold: float = 10.0,
    ) -> None:
        self.selector = selector
        self.weight_agent = weight_agent
        self.gap_threshold = gap_threshold
        self.lineage = lineage
        self.pos_label = positive_label
        self.evaluator = CrossLingualFairnessEvaluator()
        self.rl = SampleWeightRL(2, lr=rl_lr) if selector is not None else None
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    # --------------------------------------------------------------
    def update(
        self,
        stats: Dict[str, Dict[str, int]],
        dataset: str = "dataset",
        val_accuracy: float = 1.0,
    ) -> Dict[str, float]:
        """Update selection strategy using fairness ``stats`` and log results."""
        metrics = self.evaluator.evaluate(stats, positive_label=self.pos_label)
        gap = max(
            metrics.get("demographic_parity", 0.0),
            metrics.get("equal_opportunity", 0.0),
        )
        changed: Dict[str, Any] = {}
        if gap > self.gap_threshold:
            if self.weight_agent is not None:
                self.weight_agent.observe(dataset, val_accuracy, stats)
                self.weight_agent.update_db()
                changed["weight"] = self.weight_agent.weight(dataset)
            if self.selector is not None and self.rl is not None:
                probs = self.rl.weights()
                action = 0 if float(probs[0]) >= float(probs[1]) else 1
                self.rl.update(action, 1.0 - gap)
                if action == 0:
                    self.selector.threshold *= 1.1
                else:
                    self.selector.threshold *= 0.9
                self.selector.threshold = min(
                    max(self.selector.threshold, self.min_threshold),
                    self.max_threshold,
                )
                changed["threshold"] = self.selector.threshold
        if self.lineage is not None and changed:
            self.lineage.record(
                [],
                [],
                note=f"fairness_feedback metrics={metrics} changed={changed}",
            )
        return metrics


__all__ = ["FairnessFeedback"]
