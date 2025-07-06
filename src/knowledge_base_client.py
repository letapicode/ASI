from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import requests


class KnowledgeBaseClient:
    """Minimal SPARQL client with optional on-disk caching."""

    def __init__(self, endpoint: str, cache_path: str | Path | None = None) -> None:
        self.endpoint = endpoint
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, List[Tuple[str, str, str]]] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())

    # ------------------------------------------------------------
    def query(self, sparql: str) -> List[Tuple[str, str, str]]:
        if sparql in self.cache:
            return [tuple(t) for t in self.cache[sparql]]
        headers = {"Accept": "application/sparql-results+json"}
        resp = requests.post(self.endpoint, data={"query": sparql}, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for b in data.get("results", {}).get("bindings", []):
            s = b.get("s", {}).get("value")
            p = b.get("p", {}).get("value")
            o = b.get("o", {}).get("value")
            if s and p and o:
                results.append((s, p, o))
        self.cache[sparql] = results
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache))
        return results


__all__ = ["KnowledgeBaseClient"]
