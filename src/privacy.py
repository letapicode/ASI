from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional, Any

import numpy as np

try:  # pragma: no cover - allow importing standalone classes
    from .license_inspector import LicenseInspector
except Exception:  # pragma: no cover - stub for tests
    class LicenseInspector:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def inspect(self, *a: Any, **kw: Any) -> bool:
            return True

try:
    from .dataset_lineage import DatasetLineageManager
except Exception:  # pragma: no cover - stub for tests
    class DatasetLineageManager:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None:
            self.steps: list[str] = []

        def record(self, *a: Any, **kw: Any) -> None:
            self.steps.append(kw.get("note", ""))


def _write_json(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` as pretty JSON."""
    path.write_text(json.dumps(data, indent=2))


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
        _write_json(self.path, {k: asdict(v) for k, v in self.records.items()})

    # --------------------------------------------------------------
    def remaining(self, run_id: str) -> Tuple[float, float]:
        rec = self.records.get(run_id, BudgetRecord())
        return max(self.budget - rec.epsilon, 0.0), max(self.delta_budget - rec.delta, 0.0)


class PrivacyGuard:
    """Inject random noise and track a privacy budget."""

    def __init__(self, budget: float, noise_scale: float = 0.1) -> None:
        self.budget = float(budget)
        self.noise_scale = float(noise_scale)
        self._consumed = 0.0

    # --------------------------------------------------
    def _noisy_text(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        p = min(self.noise_scale, 0.5)
        kept = [w for w in words if random.random() > p]
        if not kept:
            kept = [words[0]]
        return " ".join(kept)

    def _noisy_array(self, arr: np.ndarray) -> np.ndarray:
        noise = np.random.normal(scale=self.noise_scale, size=arr.shape)
        out = arr.astype(float) + noise
        return out.astype(arr.dtype)

    # --------------------------------------------------
    def inject(
        self,
        text: str,
        image: np.ndarray,
        audio: np.ndarray,
        epsilon: float = 0.1,
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """Return noisy versions of ``text``/``image``/``audio``."""
        if self.remaining_budget() <= 0.0:
            return text, image, audio
        txt = self._noisy_text(text)
        img = self._noisy_array(np.asarray(image))
        aud = self._noisy_array(np.asarray(audio))
        self._consumed += epsilon
        return txt, img, aud

    # --------------------------------------------------
    def remaining_budget(self) -> float:
        return max(self.budget - self._consumed, 0.0)


class PrivacyAuditor:
    """Combine privacy budget tracking, license inspection and lineage logging."""

    def __init__(
        self,
        budget_manager: PrivacyBudgetManager,
        inspector: LicenseInspector,
        lineage: DatasetLineageManager,
        report_dir: str | Path = "docs/privacy_reports",
    ) -> None:
        self.budget_manager = budget_manager
        self.inspector = inspector
        self.lineage = lineage
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"compliant": 0, "rejected": 0}

    # --------------------------------------------------
    def audit_triple(
        self,
        triple: Tuple[str | Path, str | Path, str | Path],
        meta_file: str | Path | None = None,
        run_id: str = "default",
        epsilon: float = 0.01,
        delta: float = 1e-6,
    ) -> bool:
        """Audit a single triple and log budget consumption."""
        ok = True
        if meta_file is not None and Path(meta_file).exists():
            try:
                ok = self.inspector.inspect(meta_file)
            except Exception:
                ok = False
        if ok:
            self.budget_manager.consume(run_id, epsilon, delta)
            self.stats["compliant"] += 1
        else:
            self.stats["rejected"] += 1
        outs = [str(p) for p in triple]
        try:
            self.lineage.record([], outs, note=f"audit license_ok={ok}")
        except Exception:
            pass
        return ok

    # --------------------------------------------------
    def write_report(self, run_id: str, out_file: str | Path | None = None) -> Path:
        """Write a summary JSON report and return the path."""
        eps, delta = self.budget_manager.remaining(run_id)
        data = {
            "run_id": run_id,
            "remaining_epsilon": eps,
            "remaining_delta": delta,
            "license": dict(self.stats),
            "lineage_steps": len(self.lineage.steps),
        }
        if out_file is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_file = self.report_dir / f"{run_id}_{ts}.json"
        out_path = Path(out_file)
        _write_json(out_path, data)
        return out_path


__all__ = [
    "BudgetRecord",
    "PrivacyBudgetManager",
    "PrivacyGuard",
    "PrivacyAuditor",
]
