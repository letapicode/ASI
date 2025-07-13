from __future__ import annotations

from typing import Iterable, Tuple, List, Dict
import numpy as np

from .data_ingest import ActiveDataSelector
from .dataset_bias_detector import DatasetBiasDetector
from .cognitive_load_monitor import CognitiveLoadMonitor
from .dataset_lineage_manager import DatasetLineageManager
from .fairness import FairnessEvaluator


class FairnessAdaptationPipeline:
    """Adjust data selection weights using fairness metrics."""

    def __init__(
        self,
        selector: ActiveDataSelector,
        bias_detector: DatasetBiasDetector,
        load_monitor: CognitiveLoadMonitor,
        lineage: DatasetLineageManager | None = None,
    ) -> None:
        self.selector = selector
        self.bias_detector = bias_detector
        self.load_monitor = load_monitor
        self.lineage = lineage
        self.evaluator = FairnessEvaluator()

    # --------------------------------------------------------------
    def _stats(
        self,
        groups: Iterable[str],
        labels: Iterable[int],
        weights: Iterable[float] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        g_arr = np.asarray(list(groups))
        l_arr = np.asarray(list(labels), dtype=int)
        if weights is None:
            w_arr = np.ones_like(l_arr, dtype=float)
        else:
            w_arr = np.asarray(list(weights), dtype=float)
        stats: Dict[str, Dict[str, float]] = {}
        for g in np.unique(g_arr):
            mask = g_arr == g
            cnts = np.bincount(l_arr[mask], weights=w_arr[mask], minlength=2)
            d: Dict[str, float] = {}
            if cnts[0] > 0:
                d["0"] = float(cnts[0])
            if cnts[1] > 0:
                d["1"] = float(cnts[1])
            stats[g] = d
        return stats

    # --------------------------------------------------------------
    def process(
        self,
        triples: Iterable[Tuple[str, str, str]],
        probs: Iterable[np.ndarray],
        groups: Iterable[str],
        labels: Iterable[int],
    ) -> List[Tuple[Tuple[str, str, str], float]]:
        triples_list = list(triples)
        probs_list = list(probs)
        groups_list = list(groups)
        labels_list = list(labels)

        baseline = self.selector.select(triples_list, probs_list, compute_bias=False)
        stats_before = self._stats(groups_list, labels_list)
        rates = {
            g: (d.get("1", 0.0) / (sum(d.values()) + 1e-8)) for g, d in stats_before.items()
        }
        target = sum(rates.values()) / len(rates)
        load = self.load_monitor.cognitive_load()

        base_triples, base_w = zip(*baseline)
        bias_scores = np.array([
            (self.bias_detector.score_file(t[0]) + 1.0) / 2.0 for t in base_triples
        ], dtype=float)
        group_rates = np.array([rates[g] for g in groups_list], dtype=float)
        label_arr = np.asarray(labels_list, dtype=float)
        factors = np.where(
            label_arr == 1,
            target / (group_rates + 1e-8),
            (1.0 - target) / (1.0 - group_rates + 1e-8),
        )
        new_w = np.asarray(base_w, dtype=float) * factors * bias_scores * (1.0 - load)
        weighted = list(zip(base_triples, new_w.astype(float).tolist()))

        stats_after = self._stats(groups_list, labels_list, new_w)
        before = self.evaluator.evaluate(stats_before, positive_label="1")
        after = self.evaluator.evaluate(stats_after, positive_label="1")
        if self.lineage is not None:
            self.lineage.record(
                [],
                [],
                note="fairness_adaptation",
                fairness_before=before,
                fairness_after=after,
            )
        return weighted


__all__ = ["FairnessAdaptationPipeline"]
