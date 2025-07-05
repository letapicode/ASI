"""Utility to execute simple models using fully homomorphic encryption."""

from __future__ import annotations

from typing import Callable

import torch

try:
    import tenseal as ts  # type: ignore
    _HAS_TENSEAL = True
except Exception:  # pragma: no cover - optional dependency
    ts = None
    _HAS_TENSEAL = False


def run_fhe(model: Callable[[any], any], inputs: torch.Tensor, key: "ts.Context") -> torch.Tensor:
    """Encrypt ``inputs``, run ``model`` under FHE and decrypt the result."""
    if not _HAS_TENSEAL:
        raise ImportError("tenseal is required for run_fhe")

    if inputs.ndim != 1:
        raise ValueError("only 1D tensors supported")

    enc = ts.ckks_vector(key, inputs.tolist())
    out_enc = model(enc)
    if not isinstance(out_enc, ts.CKKSVector):
        raise TypeError("model must return a CKKSVector")
    out = torch.tensor(out_enc.decrypt())
    return out


__all__ = ["run_fhe"]
