from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, Tuple, Any, Dict

import numpy as np

from .cross_lingual_utils import embed_text


class CodeIndexer:
    """Stream files and embed each line deterministically."""

    def __init__(self, root: str | Path, dim: int = 32, state_path: str | Path | None = None) -> None:
        self.root = Path(root)
        self.dim = dim
        self.state_path = Path(state_path) if state_path else None
        self.embeddings: Dict[str, Dict[int, Tuple[str, np.ndarray]]] = {}
        if self.state_path and self.state_path.exists():
            data = np.load(self.state_path, allow_pickle=True).item()
            self.embeddings = data

    # --------------------------------------------------------------
    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    def _embed_line(self, line: str) -> np.ndarray:
        tokens = line.strip().split()
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        vecs = [np.asarray(embed_text(tok, self.dim)) for tok in tokens]
        arr = np.stack(vecs, axis=0)
        return arr.mean(axis=0).astype(np.float32)

    # --------------------------------------------------------------
    def index(self) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
        """Yield embeddings and metadata for changed lines."""
        for file in self.root.rglob("*"):
            if not file.is_file():
                continue
            rel = str(file.relative_to(self.root))
            stored = self.embeddings.setdefault(rel, {})
            try:
                lines = file.read_text().splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, 1):
                h = self._hash(line)
                rec = stored.get(i)
                if rec is not None and rec[0] == h:
                    continue
                vec = self._embed_line(line)
                stored[i] = (h, vec)
                meta = {"file": rel, "line": i, "text": line.rstrip()}
                yield vec, meta

    # --------------------------------------------------------------
    def save(self) -> None:
        if self.state_path:
            np.save(self.state_path, self.embeddings, allow_pickle=True)

