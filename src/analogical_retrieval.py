import torch
from typing import Tuple


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


def analogy_vector(query: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the vector for ``query : a :: ? : b`` using ``b - a``."""
    return apply_analogy(query, analogy_offset(a, b))


def analogy_search(
    memory: "HierarchicalMemory",
    query: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    k: int = 5,
    **kwargs,
):
    """Shortcut for analogy retrieval through ``HierarchicalMemory``."""
    from .hierarchical_memory import HierarchicalMemory  # circular import

    offset = analogy_offset(a, b)
    return memory.search(query, k=k, mode="analogy", offset=offset, **kwargs)


__all__ = [
    "analogy_offset",
    "apply_analogy",
    "analogy_vector",
    "analogy_search",
]
