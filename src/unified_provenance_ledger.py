from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .blockchain_provenance_ledger import BlockchainProvenanceLedger


class UnifiedProvenanceLedger:
    """Combine dataset, model and reasoning records in one blockchain ledger."""

    def __init__(self, root: str | Path) -> None:
        self.ledger = BlockchainProvenanceLedger(root)

    # --------------------------------------------------------------
    def append(self, kind: str, record: str, *, signature: Optional[str] = None) -> None:
        entry = json.dumps({"type": kind, "record": record}, sort_keys=True)
        self.ledger.append(entry, signature=signature)

    # --------------------------------------------------------------
    def append_dataset(self, record: str, *, signature: Optional[str] = None) -> None:
        self.append("dataset", record, signature=signature)

    # --------------------------------------------------------------
    def append_model(self, record: str, *, signature: Optional[str] = None) -> None:
        self.append("model", record, signature=signature)

    # --------------------------------------------------------------
    def append_reasoning(self, record: str, *, signature: Optional[str] = None) -> None:
        self.append("reasoning", record, signature=signature)

    # --------------------------------------------------------------
    def verify(self, records: Iterable[Tuple[str, str]]) -> bool:
        return self.ledger.verify(
            json.dumps({"type": t, "record": r}, sort_keys=True) for t, r in records
        )


__all__ = ["UnifiedProvenanceLedger"]
