import json
import hashlib
from pathlib import Path
from typing import Iterable, Optional

from .data_provenance_ledger import DataProvenanceLedger


class BlockchainProvenanceLedger(DataProvenanceLedger):
    """Provenance ledger implemented as a simple hash-linked blockchain."""

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "provenance_chain.jsonl"
        self.entries: list[dict[str, str]] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                self.entries.append(json.loads(line))

    # --------------------------------------------------------------
    def append(self, record: str, signature: Optional[str] = None) -> None:
        prev = self.entries[-1]["hash"] if self.entries else ""
        h = hashlib.sha256((prev + record).encode()).hexdigest()
        entry = {"hash": h, "prev": prev}
        if signature is not None:
            entry["sig"] = signature
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    # --------------------------------------------------------------
    def verify(self, records: Iterable[str]) -> bool:
        rec_list = list(records)
        if len(rec_list) != len(self.entries):
            return False
        prev = ""
        for entry, rec in zip(self.entries, rec_list):
            h = hashlib.sha256((prev + rec).encode()).hexdigest()
            if h != entry.get("hash") or entry.get("prev") != prev:
                return False
            prev = entry["hash"]
        return True


__all__ = ["BlockchainProvenanceLedger"]
