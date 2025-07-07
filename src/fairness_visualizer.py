from __future__ import annotations

import base64
import io
import json
from functools import lru_cache
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fairness_evaluator import FairnessEvaluator


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


__all__ = ["FairnessVisualizer"]
