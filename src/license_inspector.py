"""Dataset license inspection helpers."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests

from .dataset_discovery import DiscoveredDataset
from .dataset_lineage import DatasetLineageManager


# Mapping of canonical license tokens to regex patterns. Compiled once at module
# import to avoid repeated construction inside methods.
_LICENSE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("mit", re.compile(r"\bmit\b", re.I)),
    ("apache", re.compile(r"\bapache(?:[- ]?2\.0)?\b", re.I)),
    ("cc-by", re.compile(r"\bcc[- ]?by(?:-[sa0-9\.]+)?\b", re.I)),
    ("cc0", re.compile(r"\bcc0\b", re.I)),
    ("gpl", re.compile(r"\bgpl\b", re.I)),
    ("bsd", re.compile(r"\bbsd\b", re.I)),
    ("proprietary", re.compile(r"\bproprietary\b", re.I)),
)


class LicenseInspector:
    """Check dataset metadata for compatible licenses."""

    def __init__(self, allowed: Iterable[str] | None = None) -> None:
        self.allowed = {l.lower() for l in (allowed or ["mit", "apache", "cc-by"])}

    # --------------------------------------------------------------
    def _match_license(self, text: str) -> str:
        """Return a license token if ``text`` mentions a known license."""
        for token, pat in _LICENSE_PATTERNS:
            if pat.search(text):
                return token
        return ""

    def inspect(self, meta_file: str | Path) -> bool:
        """Return ``True`` if ``meta_file`` has an allowed license."""
        data = json.loads(Path(meta_file).read_text())
        lic = self._match_license(data.get("license", ""))
        return lic in self.allowed

    def inspect_dir(self, directory: str | Path) -> Dict[str, bool]:
        """Check all ``*.json`` metadata files under ``directory``."""
        out: Dict[str, bool] = {}
        for p in Path(directory).rglob("*.json"):
            out[str(p)] = self.inspect(p)
        return out

    # --------------------------------------------------------------
    def inspect_db(self, db_path: str | Path) -> Dict[str, bool]:
        """Check all datasets stored in ``db_path`` SQLite database."""
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                "SELECT name, source, license, license_text FROM datasets"
            )
            out: Dict[str, bool] = {}
            for name, src, lic, lic_text in cur:
                text = f"{lic or ''} {lic_text or ''}"
                token = self._match_license(text)
                out[f"{src}:{name}"] = token in self.allowed
        return out

    # --------------------------------------------------------------
    def report_db(self, db_path: str | Path, out_file: str | Path) -> None:
        """Write a JSON report for datasets in ``db_path``."""
        res = self.inspect_db(db_path)
        Path(out_file).write_text(json.dumps(res, indent=2))

    # --------------------------------------------------------------
    def inspect_discovered(
        self,
        dataset: DiscoveredDataset,
        lineage: DatasetLineageManager | None = None,
    ) -> bool:
        """Check a :class:`DiscoveredDataset` and optionally log to lineage."""
        text = f"{dataset.license} {dataset.license_text}"
        if not text.strip() and dataset.url:
            try:
                resp = requests.get(dataset.url, timeout=5)
                resp.raise_for_status()
                text = resp.text
            except Exception:
                text = ""
        lic = self._match_license(text)
        allowed = lic in self.allowed
        if lineage is not None:
            note = f"inspect {dataset.source}:{dataset.name} license={lic or 'unknown'} allowed={allowed}"
            try:
                lineage.record([dataset.url], [], note=note)
            except Exception:
                pass
        return allowed

    # --------------------------------------------------------------
    def scan_discovered_db(
        self,
        db_path: str | Path,
        lineage: DatasetLineageManager | None = None,
    ) -> Dict[str, bool]:
        """Inspect all discovered datasets in ``db_path``."""
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                "SELECT name, source, url, license, license_text FROM datasets"
            )
            out: Dict[str, bool] = {}
            for name, source, url, lic, lic_text in cur:
                d = DiscoveredDataset(name, url, source, lic or "", lic_text or "")
                ok = self.inspect_discovered(d, lineage)
                out[f"{source}:{name}"] = ok
        return out


__all__ = ["LicenseInspector"]
