"""Cross-modal analogy retrieval utilities."""

from typing import Any, Tuple, List

import torch

from .cross_modal_fusion import CrossModalFusion, MultiModalDataset, encode_all
from .analogical_retrieval import analogy_offset
from .hierarchical_memory import HierarchicalMemory


def cross_modal_analogy_search(
    model: CrossModalFusion,
    dataset: MultiModalDataset,
    memory: HierarchicalMemory,
    query_index: int,
    a_index: int,
    b_index: int,
    *,
    k: int = 5,
    batch_size: int = 8,
    **kwargs: Any,
) -> Tuple[torch.Tensor, List[Any]]:
    """Return analogy search results across modalities using ``memory``.

    The dataset is encoded via :func:`encode_all` and stored in ``memory``. The
    fused embeddings (average of text, image and audio) are used to compute the
    offset ``b - a`` and query ``query + offset``.
    """

    t_vecs, i_vecs, a_vecs = encode_all(
        model, dataset, batch_size=batch_size, memory=memory, **kwargs
    )
    fused = (t_vecs + i_vecs + a_vecs) / 3.0

    offset = analogy_offset(fused[a_index], fused[b_index])
    return memory.search(fused[query_index], k=k, mode="analogy", offset=offset)


__all__ = ["cross_modal_analogy_search"]
