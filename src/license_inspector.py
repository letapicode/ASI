"""Dataset license inspection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict


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


__all__ = ["LicenseInspector"]
