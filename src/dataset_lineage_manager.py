from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict


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
    outputs: Dict[str, str]


class DatasetLineageManager:
    """Record dataset transformations and resulting file hashes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.log_path = self.root / "dataset_lineage.json"
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
    ) -> None:
        out_hashes = {str(p): _hash_file(Path(p)) for p in outputs}
        step = LineageStep(note, [str(p) for p in inputs], out_hashes)
        self.steps.append(step)
        self.log_path.write_text(
            json.dumps([asdict(s) for s in self.steps], indent=2)
        )

    # --------------------------------------------------------------
    def load(self) -> List[LineageStep]:
        """Reload steps from the log file."""
        if self.log_path.exists():
            data = json.loads(self.log_path.read_text())
            self.steps = [LineageStep(**d) for d in data]
        return self.steps


__all__ = ["DatasetLineageManager", "LineageStep"]
