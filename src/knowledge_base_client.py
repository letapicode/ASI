from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import requests
import yaml


class KnowledgeBaseClient:
    """Minimal client for SPARQL knowledge bases."""

    def __init__(self, endpoint: str, cache_size: int = 0) -> None:
        self.endpoint = endpoint
        self.cache_size = int(cache_size)
        self.cache: OrderedDict[str, List[Tuple[str, str, str]]] = OrderedDict()

    # --------------------------------------------------------------
    def query(self, sparql: str) -> List[Tuple[str, str, str]]:
        """Return triples for ``sparql`` query."""
        if sparql in self.cache:
            self.cache.move_to_end(sparql)
            return self.cache[sparql]
        resp = requests.post(
            self.endpoint,
            data={"query": sparql},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        out: List[Tuple[str, str, str]] = []
        for b in data.get("results", {}).get("bindings", []):
            s = b.get("s", {}).get("value", "")
            p = b.get("p", {}).get("value", "")
            o = b.get("o", {}).get("value", "")
            out.append((s, p, o))
        if self.cache_size > 0:
            self.cache[sparql] = out
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return out


def load_kb_config(path: str | Path) -> dict[str, object]:
    """Load knowledge base settings from ``path``."""
    file = Path(path)
    if not file.exists():
        return {}
    data = yaml.safe_load(file.read_text()) or {}
    return {
        "endpoint": str(data.get("endpoint", "")),
        "cache_size": int(data.get("cache_size", 0)),
    }


__all__ = ["KnowledgeBaseClient", "load_kb_config"]
