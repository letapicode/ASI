"""Generate weak labels for unlabeled triples using a world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import torch

from .multimodal_world_model import MultiModalWorldModel


Tokenizer = Callable[[str], Sequence[int]]


@dataclass
class AutoLabelerConfig:
    """Configuration for :class:`AutoLabeler`."""

    num_labels: int
    tokenizer: Tokenizer | None = None


class AutoLabeler:
    """Use a trained world model to assign weak labels."""

    def __init__(self, model: MultiModalWorldModel, cfg: AutoLabelerConfig) -> None:
        self.model = model
        self.cfg = cfg
        tok = cfg.tokenizer
        if tok is None:
            size = model.cfg.vocab_size

            def tok(text: str) -> Sequence[int]:
                return [ord(c) % size for c in text]

        self.tokenizer = tok
        self.classifier = torch.nn.Linear(model.cfg.embed_dim, cfg.num_labels)

    def _prep_batch(self, triples: Iterable[Tuple[str, np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        texts, imgs, acts = zip(*triples)
        tokens = [torch.tensor(self.tokenizer(t), dtype=torch.long) for t in texts]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        images = torch.tensor(np.stack(imgs), dtype=torch.float32)
        actions = torch.tensor(acts, dtype=torch.long)
        return tokens, images, actions

    def label(self, triples: Iterable[Tuple[str, np.ndarray, int]]) -> list[int]:
        tokens, images, actions = self._prep_batch(triples)
        device = next(self.model.parameters()).device
        tokens = tokens.to(device)
        images = images.to(device)
        actions = actions.to(device)
        self.model.eval()
        with torch.no_grad():
            state = self.model.encode_obs(tokens, images)
            logits = self.classifier(state)
            preds = torch.argmax(logits, dim=-1)
        return preds.cpu().tolist()


__all__ = ["AutoLabeler", "AutoLabelerConfig"]
