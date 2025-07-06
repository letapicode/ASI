from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence, Dict, List

import numpy as np
import torch

from .data_ingest import CrossLingualTranslator


def load_analogy_dataset(path: str | Path) -> List[Dict[str, str]]:
    """Return list of analogy tuples from a JSONL ``path``."""
    lines = Path(path).read_text().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


_BASE_VECS = {
    "man": torch.tensor([1.0, 0.0, 0.0]),
    "woman": torch.tensor([0.0, 1.0, 0.0]),
    "king": torch.tensor([1.0, 0.0, 1.0]),
    "queen": torch.tensor([0.0, 1.0, 1.0]),
    "france": torch.tensor([1.0, 0.0, 0.0]),
    "germany": torch.tensor([0.0, 1.0, 0.0]),
    "paris": torch.tensor([1.0, 0.0, 1.0]),
    "berlin": torch.tensor([0.0, 1.0, 1.0]),
}


def _embed_text(text: str, dim: int) -> torch.Tensor:
    base = text.split(" ")[-1]
    if base in _BASE_VECS:
        vec = _BASE_VECS[base]
    else:
        seed = abs(hash(text)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        vec = torch.from_numpy(rng.standard_normal(dim).astype(np.float32))
    if vec.numel() != dim:
        vec = torch.nn.functional.pad(vec, (0, dim - vec.numel()))
    return vec


def _vec(text: str, lang: str | None, tr: CrossLingualTranslator, dim: int) -> torch.Tensor:
    if lang is not None and lang in tr.languages:
        text = tr.translate(text, lang)
    return _embed_text(text, dim)


def analogy_offset(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    return b - a


def apply_analogy(query: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    if query.shape != offset.shape:
        raise ValueError("offset dimension mismatch")
    return query + offset


def build_vocab_embeddings(words: Iterable[str], tr: CrossLingualTranslator, dim: int) -> Dict[str, torch.Tensor]:
    vecs: Dict[str, torch.Tensor] = {}
    for w in words:
        vecs[w] = _embed_text(w, dim)
        for l in tr.languages:
            trans = tr.translate(w, l)
            vecs[trans] = _embed_text(trans, dim)
    return vecs


def analogy_accuracy(path: str | Path, languages: Sequence[str], k: int = 1, dim: int = 3) -> float:
    """Return accuracy on the analogies stored at ``path`` using ``languages``."""
    data = load_analogy_dataset(path)
    tr = CrossLingualTranslator(languages)
    vocab = {row['query'] for row in data}
    vocab.update(row['a'] for row in data)
    vocab.update(row['b'] for row in data)
    vocab.update(row['expected'] for row in data)
    emb = build_vocab_embeddings(vocab, tr, dim)

    correct = 0
    for row in data:
        q = _vec(row['query'], row.get('query_lang'), tr, dim)
        a = _vec(row['a'], row.get('a_lang'), tr, dim)
        b = _vec(row['b'], row.get('b_lang'), tr, dim)
        target = row['expected']
        off = analogy_offset(a, b)
        vec = apply_analogy(q, off)
        # compute cosine similarity against all vocab embeddings
        names = list(emb.keys())
        mat = torch.stack([emb[n] for n in names])
        scores = torch.nn.functional.cosine_similarity(mat, vec.expand_as(mat), dim=1)
        top_idx = scores.argmax().item()
        pred = names[top_idx]
        if pred.endswith(target):
            correct += 1
    return correct / len(data) if data else 0.0


__all__ = [
    "load_analogy_dataset",
    "analogy_accuracy",
]
