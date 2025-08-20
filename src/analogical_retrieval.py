import json
from pathlib import Path
from typing import Tuple, List, Any, TYPE_CHECKING, Iterable, Sequence, Dict

import torch

from .cross_lingual_translator import CrossLingualTranslator
from .cross_lingual_utils import embed_text

if TYPE_CHECKING:  # pragma: no cover - import for type hints
    from .hierarchical_memory import HierarchicalMemory


def _to_vec(
    item: torch.Tensor | str | Tuple[str, str],
    memory: "HierarchicalMemory",
    translator: CrossLingualTranslator | None,
    default_lang: str | None,
) -> torch.Tensor:
    """Return embedding for ``item`` using ``translator`` if needed."""
    if isinstance(item, torch.Tensor):
        return item
    text: str
    lang: str | None = default_lang
    if isinstance(item, tuple):
        text, lang = item
    else:
        text = item
    if (
        translator is not None
        and lang is not None
        and lang in getattr(translator, "languages", [])
    ):
        text = translator.translate(text, lang)
    dim = memory.compressor.encoder.in_features
    return _embed_text(text, dim)


def analogy_offset(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the offset vector ``b - a`` used for analogy arithmetic."""
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    return b - a


def apply_analogy(query: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Apply ``offset`` to ``query`` (i.e. ``query + offset``)."""
    if query.shape != offset.shape:
        raise ValueError("offset dimension mismatch")
    return query + offset


def analogy_vector(
    query: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Return the vector for ``query : a :: ? : b`` using ``b - a``."""
    return apply_analogy(query, analogy_offset(a, b))


def analogy_search(
    memory: "HierarchicalMemory",
    query: torch.Tensor | str | Tuple[str, str],
    a: torch.Tensor | str | Tuple[str, str],
    b: torch.Tensor | str | Tuple[str, str],
    k: int = 5,
    *,
    language: str | None = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, List[Any]]:
    """Shortcut for analogy retrieval through ``HierarchicalMemory``.

    ``query``, ``a`` and ``b`` may be strings or ``(text, lang)`` tuples.
    If a ``CrossLingualTranslator`` is attached to ``memory`` the strings are
    translated before embedding.
    """

    translator = getattr(memory, "translator", None)
    q_vec = _to_vec(query, memory, translator, language)
    a_vec = _to_vec(a, memory, translator, language)
    b_vec = _to_vec(b, memory, translator, language)
    offset = analogy_offset(a_vec, b_vec)
    try:
        return memory.search(
            q_vec,
            k=k,
            mode="analogy",
            offset=offset,
            language=language,
            **kwargs,
        )
    except TypeError:
        return memory.search(q_vec + offset, k=k, **kwargs)


def _dataset_vec(
    text: str, lang: str | None, tr: CrossLingualTranslator, dim: int
) -> torch.Tensor:
    if lang is not None and lang in tr.languages:
        text = tr.translate(text, lang)
    return embed_text(text, dim)


def load_analogy_dataset(path: str | Path) -> List[Dict[str, str]]:
    """Return list of analogy tuples from a JSONL ``path``."""
    lines = Path(path).read_text().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def build_vocab_embeddings(
    words: Iterable[str], tr: CrossLingualTranslator, dim: int
) -> Dict[str, torch.Tensor]:
    vecs: Dict[str, torch.Tensor] = {}
    for w in words:
        vecs[w] = embed_text(w, dim)
        for l in tr.languages:
            trans = tr.translate(w, l)
            vecs[trans] = embed_text(trans, dim)
    return vecs


def analogy_accuracy(
    path: str | Path, languages: Sequence[str], k: int = 1, dim: int = 3
) -> float:
    """Return accuracy on the analogies stored at ``path`` using ``languages``."""
    data = load_analogy_dataset(path)
    tr = CrossLingualTranslator(languages)
    vocab = {row["query"] for row in data}
    vocab.update(row["a"] for row in data)
    vocab.update(row["b"] for row in data)
    vocab.update(row["expected"] for row in data)
    emb = build_vocab_embeddings(vocab, tr, dim)

    correct = 0
    for row in data:
        q = _dataset_vec(row["query"], row.get("query_lang"), tr, dim)
        a = _dataset_vec(row["a"], row.get("a_lang"), tr, dim)
        b = _dataset_vec(row["b"], row.get("b_lang"), tr, dim)
        target = row["expected"]
        off = analogy_offset(a, b)
        vec = apply_analogy(q, off)
        names = list(emb.keys())
        mat = torch.stack([emb[n] for n in names])
        scores = torch.nn.functional.cosine_similarity(
            mat, vec.expand_as(mat), dim=1
        )
        pred = names[scores.argmax().item()]
        if pred.endswith(target):
            correct += 1
    return correct / len(data) if data else 0.0


__all__ = [
    "analogy_offset",
    "apply_analogy",
    "analogy_vector",
    "analogy_search",
    "load_analogy_dataset",
    "build_vocab_embeddings",
    "analogy_accuracy",
]
