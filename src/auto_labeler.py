"""Generate weak labels for unlabeled triples using a world model."""

from __future__ import annotations

from typing import Iterable, Tuple, Any

import numpy as np
import torch

from .multimodal_world_model import MultiModalWorldModel


class AutoLabeler:
    """Weakly label samples by embedding them with a world model."""

    def __init__(self, model: MultiModalWorldModel, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def _hash_label(self, text: str) -> int:
        vocab = self.model.cfg.vocab_size
        return sum(ord(c) for c in text) % vocab

    def label(self, triples: Iterable[Tuple[str, np.ndarray, Any | None]]) -> list[int]:
        device = next(self.model.parameters()).device
        labels = []
        for text, img, _ in triples:
            try:
                t = torch.tensor(self.tokenizer(text), dtype=torch.long, device=device).unsqueeze(0)
                im = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
                state = self.model.encode_obs(t, im)
                val = int(state.mean().item() * 1000) % self.model.cfg.vocab_size
            except Exception:
                val = self._hash_label(text)
            labels.append(val)
        return labels


__all__ = ["AutoLabeler"]
