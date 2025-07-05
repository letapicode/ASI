import torch
from typing import Tuple, List, Any, TYPE_CHECKING

from .data_ingest import CrossLingualTranslator
from .cross_lingual_memory import _embed_text

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
    if translator is not None and lang is not None:
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
    return memory.search(
        q_vec, k=k, mode="analogy", offset=offset, language=language, **kwargs
    )


__all__ = [
    "analogy_offset",
    "apply_analogy",
    "analogy_vector",
    "analogy_search",
]
