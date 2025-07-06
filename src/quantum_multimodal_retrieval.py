"""Quantum-assisted retrieval over multimodal embeddings."""

from __future__ import annotations

from typing import Tuple, Any

try:  # optional torch dependency
    import torch
except Exception:  # pragma: no cover - allow running without torch
    import numpy as np
    import types

    class _DummyTensor(np.ndarray):
        def to(self, device=None):
            return self

        @property
        def device(self):
            return "cpu"

    def _from_numpy(arr):
        return np.asarray(arr, dtype=np.float32).view(_DummyTensor)

    def _stack(seq, dim=0):
        return np.stack(seq, axis=dim).view(_DummyTensor)

    class _DummyTorch(types.SimpleNamespace):
        pass

    torch = _DummyTorch(
        from_numpy=_from_numpy,
        stack=_stack,
        empty=lambda *shape, device=None: np.empty(shape, dtype=np.float32).view(_DummyTensor),
    )

from .quantum_retrieval import amplify_search


def _fuse(text: torch.Tensor | None, image: torch.Tensor | None, audio: torch.Tensor | None) -> torch.Tensor:
    """Return averaged embedding over available modalities."""
    parts = [p for p in (text, image, audio) if p is not None]
    if not parts:
        raise ValueError("at least one modality must be provided")
    if len(parts) == 1:
        return parts[0]
    stack = torch.stack(parts)
    return stack.mean(dim=0)


def quantum_crossmodal_search(
    query: Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | torch.Tensor,
    memory: Any,
    k: int = 5,
) -> Tuple[torch.Tensor, list[Any]]:
    """Search ``memory`` using amplitude amplification over fused modalities."""
    if isinstance(query, tuple):
        text, img, aud = query
        q = _fuse(text, img, aud)
    else:
        q = query
    device = q.device
    vecs, meta, scores = memory.search(q, k=len(memory), return_scores=True)
    if not scores:
        return vecs, meta
    order = amplify_search(scores, k)
    out_vecs = vecs[order]
    out_meta = [meta[i] for i in order]
    return out_vecs.to(device), out_meta


__all__ = ["quantum_crossmodal_search"]
