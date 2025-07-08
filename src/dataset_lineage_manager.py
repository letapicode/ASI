from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, List, Dict

from .dataset_watermarker import detect_watermark

from .data_provenance_ledger import DataProvenanceLedger


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LineageStep:
    note: str
    inputs: List[str]
    outputs: Dict[str, Dict[str, str | None]]
    fairness_before: Dict[str, float] | None = None
    fairness_after: Dict[str, float] | None = None


class DatasetLineageManager:
    """Record dataset transformations and resulting file hashes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.log_path = self.root / "dataset_lineage.json"
        self.ledger = DataProvenanceLedger(self.root)
        if self.log_path.exists():
            data = json.loads(self.log_path.read_text())
            self.steps: List[LineageStep] = [LineageStep(**d) for d in data]
        else:
            self.steps = []

    # --------------------------------------------------------------
    def record(
        self,
        inputs: Iterable[str | Path],
        outputs: Iterable[str | Path],
        note: str = "",
        fairness_before: Dict[str, float] | None = None,
        fairness_after: Dict[str, float] | None = None,
    ) -> None:
        out_hashes: Dict[str, Dict[str, str | None]] = {}
        for p in outputs:
            path = Path(p)
            out_hashes[str(path)] = {
                "hash": _hash_file(path),
                "watermark_id": detect_watermark(path),
            }
        step = LineageStep(
            note,
            [str(p) for p in inputs],
            out_hashes,
            fairness_before,
            fairness_after,
        )
        self.steps.append(step)
        self.log_path.write_text(
            json.dumps([asdict(s) for s in self.steps], indent=2)
        )
        rec = json.dumps(asdict(step), sort_keys=True)
        self.ledger.append(rec)

    # --------------------------------------------------------------
    def load(self) -> List[LineageStep]:
        """Reload steps from the log file."""
        if self.log_path.exists():
            data = json.loads(self.log_path.read_text())
            self.steps = [LineageStep(**d) for d in data]
        return self.steps


__all__ = ["DatasetLineageManager", "LineageStep"]
