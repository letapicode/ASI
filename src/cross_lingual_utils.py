import numpy as np

try:  # optional torch dependency
    import torch
except Exception:  # pragma: no cover - allow running without torch
    from .torch_fallback import torch

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

def embed_text(text: str, dim: int):
    """Deterministically embed ``text`` with simple seed vectors."""
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

__all__ = ["embed_text"]
