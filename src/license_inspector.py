"""Dataset license inspection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict
import sqlite3


class LicenseInspector:
    """Check dataset metadata for compatible licenses."""

    def __init__(self, allowed: Iterable[str] | None = None) -> None:
        self.allowed = {l.lower() for l in (allowed or ["mit", "apache", "cc-by"])}

    def inspect(self, meta_file: str | Path) -> bool:
        """Return ``True`` if ``meta_file`` has an allowed license."""
        data = json.loads(Path(meta_file).read_text())
        lic = data.get("license", "").lower()
        return any(a in lic for a in self.allowed)

    def inspect_dir(self, directory: str | Path) -> Dict[str, bool]:
        """Check all ``*.json`` metadata files under ``directory``."""
        out: Dict[str, bool] = {}
        for p in Path(directory).rglob("*.json"):
            out[str(p)] = self.inspect(p)
        return out

    # --------------------------------------------------------------
    def inspect_db(self, db_path: str | Path) -> Dict[str, bool]:
        """Check all datasets stored in ``db_path`` SQLite database."""
        conn = sqlite3.connect(db_path)
        cur = conn.execute(
            "SELECT name, source, license, license_text FROM datasets"
        )
        out: Dict[str, bool] = {}
        for name, src, lic, lic_text in cur:
            text = (lic or "") + " " + (lic_text or "")
            text = text.lower()
            out[f"{src}:{name}"] = any(a in text for a in self.allowed)
        conn.close()
        return out

    # --------------------------------------------------------------
    def report_db(self, db_path: str | Path, out_file: str | Path) -> None:
        """Write a JSON report for datasets in ``db_path``."""
        res = self.inspect_db(db_path)
        Path(out_file).write_text(json.dumps(res, indent=2))


__all__ = ["LicenseInspector"]
