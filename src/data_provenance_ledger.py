import json
import hashlib
from pathlib import Path
from typing import Iterable, Optional


class DataProvenanceLedger:
    """Append hashed (and optionally signed) lineage records."""

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "provenance_ledger.jsonl"
        self.entries: list[dict[str, str]] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                self.entries.append(json.loads(line))

    def append(self, record: str, signature: Optional[str] = None) -> None:
        h = hashlib.sha256(record.encode()).hexdigest()
        entry = {"hash": h}
        if signature is not None:
            entry["sig"] = signature
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def verify(self, records: Iterable[str]) -> bool:
        rec_list = list(records)
        if len(rec_list) != len(self.entries):
            return False
        for entry, rec in zip(self.entries, rec_list):
            h = hashlib.sha256(rec.encode()).hexdigest()
            if h != entry.get("hash"):
                return False
        return True

__all__ = ["DataProvenanceLedger"]
