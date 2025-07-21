from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple, List

import numpy as np

from .data_ingest import ActiveDataSelector
from .dataset_bias_detector import DatasetBiasDetector
from .cognitive_load_monitor import CognitiveLoadMonitor
from .dataset_lineage_manager import DatasetLineageManager
from pathlib import Path
import sys

pkg = sys.modules.get('asi')
if pkg is not None and not getattr(pkg, '__path__', None):
    pkg.__path__ = [str(Path(__file__).parent)]

# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency
    from .data_ingest import CrossLingualTranslator
except Exception:  # pragma: no cover - missing torch
    from .translator_fallback import CrossLingualTranslator


class CrossLingualFairnessEvaluator:
    """FairnessEvaluator that normalizes groups across languages."""

    def __init__(
        self,
        translator: CrossLingualTranslator | None = None,
        target_lang: str = "en",
    ) -> None:
        self.translator = translator
        self.target_lang = target_lang
        self.base = FairnessEvaluator()

    def _translate(self, text: str) -> str:
        if self.translator is None:
            return text
        return self.translator.translate(text, self.target_lang)

    def _normalize(self, stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        norm: Dict[str, Dict[str, int]] = {}
        for group, counts in stats.items():
            g = self._translate(group)
            out = norm.setdefault(g, {})
            for label, cnt in counts.items():
                if label in {"tp", "fp", "fn", "tn"}:
                    l = label
                else:
                    l = self._translate(label)
                out[l] = out.get(l, 0) + cnt
        return norm

    def evaluate(self, stats: Dict[str, Dict[str, int]], positive_label: str = "1") -> Dict[str, float]:
        norm = self._normalize(stats)
        if positive_label in {"tp", "fp", "fn", "tn"}:
            pos = positive_label
        else:
            pos = self._translate(positive_label)
        return self.base.evaluate(norm, positive_label=pos)


# ---------------------------------------------------------------------------
DatasetWeightAgent = None
ActiveDataSelector = None
SampleWeightRL = None
DatasetLineageManager = None


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
        global DatasetWeightAgent, ActiveDataSelector, SampleWeightRL, DatasetLineageManager
        if DatasetWeightAgent is None:
            try:
                from .dataset_weight_agent import DatasetWeightAgent as _DWA
            except Exception:  # pragma: no cover - during tests
                from dataset_weight_agent import DatasetWeightAgent as _DWA  # type: ignore
            DatasetWeightAgent = _DWA
        if ActiveDataSelector is None:
            try:
                from .data_ingest import ActiveDataSelector as _ADS
            except Exception:  # pragma: no cover - during tests
                from data_ingest import ActiveDataSelector as _ADS  # type: ignore
            ActiveDataSelector = _ADS
        if SampleWeightRL is None:
            try:
                from .adaptive_curriculum import SampleWeightRL as _SWRL
            except Exception:  # pragma: no cover - during tests
                from adaptive_curriculum import SampleWeightRL as _SWRL  # type: ignore
            SampleWeightRL = _SWRL
        if DatasetLineageManager is None:
            try:
                from .dataset_lineage_manager import DatasetLineageManager as _DLM
            except Exception:  # pragma: no cover - during tests
                from dataset_lineage_manager import DatasetLineageManager as _DLM  # type: ignore
            DatasetLineageManager = _DLM

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


# ---------------------------------------------------------------------------
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
        bias_scores = np.array(
            [(self.bias_detector.score_file(t[0]) + 1.0) / 2.0 for t in base_triples],
            dtype=float,
        )
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


# ---------------------------------------------------------------------------
import base64
import io
import json
import matplotlib
from functools import lru_cache
from typing import Dict, Any

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class FairnessVisualizer:
    """Render demographic parity and equal opportunity gaps.

    Caches previously generated images keyed by the input stats so repeated
    calls avoid redundant plotting work.
    """

    def __init__(self, evaluator: FairnessEvaluator | None = None) -> None:
        self.evaluator = evaluator or FairnessEvaluator()
        self._cache: Dict[str, str] = {}

    # --------------------------------------------------------------
    def _bar_plot(self, labels: list[str], dp: list[float], eo: list[float]) -> str:
        fig, ax = plt.subplots(figsize=(max(2, len(labels)), 2))
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], dp, width=0.4, label="parity")
        ax.bar([i + 0.2 for i in x], eo, width=0.4, label="opportunity")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("gap")
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # --------------------------------------------------------------
    def to_image(self, results: Dict[str, Any], positive_label: str = "1") -> str:
        """Return a base64 PNG for ``results`` from ``FairnessEvaluator``."""
        key = json.dumps(results, sort_keys=True) + positive_label
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # ``results`` can be raw stats or already-computed metrics.
        if "demographic_parity" in results and "equal_opportunity" in results:
            labels = ["dataset"]
            dp_vals = [float(results["demographic_parity"])]
            eo_vals = [float(results["equal_opportunity"])]
            img = self._bar_plot(labels, dp_vals, eo_vals)
            self._cache[key] = img
            return img

        # maybe multimodal metrics
        first = next(iter(results.values()))
        if isinstance(first, dict) and "demographic_parity" in first:
            labels = []
            dp_vals = []
            eo_vals = []
            for mod, vals in results.items():
                labels.append(mod)
                dp_vals.append(float(vals.get("demographic_parity", 0.0)))
                eo_vals.append(float(vals.get("equal_opportunity", 0.0)))
            img = self._bar_plot(labels, dp_vals, eo_vals)
            self._cache[key] = img
            return img

        # assume raw stats mapping group->counts or modality->group->counts
        if isinstance(first, dict) and any(isinstance(v, dict) for v in first.values()):
            metrics = self.evaluator.evaluate_multimodal(results, positive_label)
            img = self.to_image(metrics)
            self._cache[key] = img
            return img
        metrics = self.evaluator.evaluate(results, positive_label)
        img = self.to_image(metrics)
        self._cache[key] = img
        return img


__all__ = [
    "FairnessEvaluator",
    "CrossLingualFairnessEvaluator",
    "FairnessFeedback",
    "FairnessAdaptationPipeline",
    "FairnessVisualizer",
]
