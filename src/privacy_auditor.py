from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple, Optional

from .privacy_budget_manager import PrivacyBudgetManager
from .license_inspector import LicenseInspector
from .dataset_lineage_manager import DatasetLineageManager


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
        out_path.write_text(json.dumps(data, indent=2))
        return out_path


__all__ = ["PrivacyAuditor"]
