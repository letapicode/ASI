from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Iterable

from .dataset_lineage import DatasetLineageManager, LineageStep
from .blockchain_provenance_ledger import BlockchainProvenanceLedger


class RetrievalTrustScorer:
    """Score retrieval results based on dataset lineage and provenance ledger."""

    def __init__(
        self,
        lineage: DatasetLineageManager | None = None,
        ledger: BlockchainProvenanceLedger | None = None,
    ) -> None:
        self.lineage = lineage
        self.ledger = ledger
        self._output_map: dict[str, tuple[int, LineageStep]] = {}
        if lineage is not None:
            self._index_lineage()
        self._ledger_hashes: dict[int, str] = {}
        if ledger is not None:
            self._index_ledger()

    # --------------------------------------------------------------
    def _index_lineage(self) -> None:
        """Precompute a mapping from output paths to steps."""
        assert self.lineage is not None
        for idx, step in enumerate(self.lineage.steps):
            for out in step.outputs:
                self._output_map[out] = (idx, step)

    # --------------------------------------------------------------
    def _index_ledger(self) -> None:
        """Precompute ledger entry hashes for quick lookup."""
        assert self.ledger is not None
        for idx, entry in enumerate(self.ledger.entries):
            if "hash" in entry:
                self._ledger_hashes[idx] = entry["hash"]

    # --------------------------------------------------------------
    def _step_for_output(self, path: str) -> tuple[int, LineageStep] | None:
        if self.lineage is None:
            return None
        if path in self._output_map:
            return self._output_map[path]
        # Fallback to linear search for unknown entries
        for idx, step in enumerate(self.lineage.steps):
            if path in step.outputs:
                self._output_map[path] = (idx, step)
                return idx, step
        return None

    # --------------------------------------------------------------
    def trust_score(self, meta: Any) -> float:
        path = None
        if isinstance(meta, dict):
            path = meta.get("path") or meta.get("file") or meta.get("source")
        else:
            path = str(meta)
        score = 0.0
        if path:
            info = self._step_for_output(path)
            if info is not None:
                idx, step = info
                score += 0.5
                if self.ledger is not None:
                    if idx not in self._ledger_hashes:
                        rec = json.dumps(asdict(step), sort_keys=True)
                        h = hashlib.sha256(rec.encode()).hexdigest()
                        self._ledger_hashes[idx] = h
                    if self._ledger_hashes.get(idx) == self.ledger.entries[idx].get("hash"):
                        score += 0.5
        return score

    # --------------------------------------------------------------
    def score_results(self, provenance: Iterable[Any]) -> list[float]:
        """Return trust scores for ``provenance`` entries."""
        return [self.trust_score(p) for p in provenance]


__all__ = ["RetrievalTrustScorer"]
