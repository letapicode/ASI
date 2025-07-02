"""Track dataset versions and transformation steps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class DatasetVersioner:
    """Write a version file with hashes of all stored data."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def record(self, files: Iterable[str | Path], note: str = "") -> None:
        entries = {str(p): _hash_file(Path(p)) for p in files}
        data = {"note": note, "files": entries}
        out = self.root / "dataset_version.json"
        out.write_text(json.dumps(data, indent=2))


__all__ = ["DatasetVersioner"]
