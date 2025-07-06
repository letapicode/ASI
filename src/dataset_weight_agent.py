from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

try:
    from .dataset_bias_detector import text_bias_score
except Exception:  # pragma: no cover - during tests
    from dataset_bias_detector import text_bias_score  # type: ignore

try:  # pragma: no cover - for standalone imports
    from .license_inspector import LicenseInspector
except Exception:  # pragma: no cover - during tests
    from license_inspector import LicenseInspector  # type: ignore

try:
    from .fairness_evaluator import FairnessEvaluator
except Exception:  # pragma: no cover - during tests
    from fairness_evaluator import FairnessEvaluator  # type: ignore


class DatasetWeightAgent:
    """Maintain dataset weights using bias and license signals."""

    def __init__(
        self,
        db_path: str | Path,
        allowed_licenses: Iterable[str] | None = None,
        epsilon: float = 0.1,
        alpha: float = 0.5,
        gamma: float = 0.9,
    ) -> None:
        self.db_path = Path(db_path)
        self.inspector = LicenseInspector(allowed_licenses)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.q: Dict[Tuple[str, str], float] = {}
        self.weights: Dict[str, float] = {}
        self._load()

    # --------------------------------------------------------------
    def _load(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            "SELECT name, source, license, license_text FROM datasets"
        )
        for name, src, lic, lic_text in cur:
            key = f"{src}:{name}"
            lic_ok = any(
                a in (lic or "" + " " + (lic_text or "")).lower()
                for a in self.inspector.allowed
            )
            bias = text_bias_score(name)
            self.weights[key] = float(bias if lic_ok else 0.0)
        conn.close()

    # --------------------------------------------------------------
    def sample(self, k: int = 1) -> list[str]:
        names = list(self.weights.keys())
        if not names:
            return []
        w = np.array([self.weights[n] for n in names], dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        p = w / w.sum()
        idx = np.random.choice(len(names), size=min(k, len(names)), replace=False, p=p)
        return [names[i] for i in idx]

    # --------------------------------------------------------------
    def weight(self, name: str) -> float:
        return float(self.weights.get(name, 0.0))

    # --------------------------------------------------------------
    def observe(
        self,
        dataset: str,
        val_accuracy: float,
        fairness_stats: Dict[str, Dict[str, int]],
    ) -> None:
        evaluator = FairnessEvaluator()
        scores = evaluator.evaluate(fairness_stats)
        fairness = 1.0 - (
            scores.get("demographic_parity", 0.0)
            + scores.get("equal_opportunity", 0.0)
        ) / 2.0
        reward = (float(val_accuracy) + fairness) / 2.0
        inc_key = (dataset, "inc")
        dec_key = (dataset, "dec")
        if random.random() < self.epsilon:
            action = random.choice(["inc", "dec"])
        else:
            inc_q = self.q.get(inc_key, 0.0)
            dec_q = self.q.get(dec_key, 0.0)
            action = "inc" if inc_q >= dec_q else "dec"
        current = self.q.get((dataset, action), 0.0)
        next_q = max(self.q.get(inc_key, 0.0), self.q.get(dec_key, 0.0))
        target = reward + self.gamma * next_q
        self.q[(dataset, action)] = current + self.alpha * (target - current)
        if action == "inc":
            self.weights[dataset] = min(1.0, self.weights.get(dataset, 0.0) + 0.1)
        else:
            self.weights[dataset] = max(0.0, self.weights.get(dataset, 0.0) - 0.1)

    # --------------------------------------------------------------
    def update_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        rows = []
        for key, w in self.weights.items():
            try:
                src, name = key.split(":", 1)
            except ValueError:
                continue
            rows.append((float(w), name, src))
        with conn:
            conn.executemany(
                "UPDATE datasets SET weight=? WHERE name=? AND source=?",
                rows,
            )
        conn.close()


__all__ = ["DatasetWeightAgent"]
