from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Any

import random
import torch


@dataclass
class IngestConfig:
    crop_size: int = 32
    noise_std: float = 0.01


def align_modalities(
    texts: Iterable[str], images: Iterable[torch.Tensor], audios: Iterable[torch.Tensor]
) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Return paired text, image and audio items."""
    triples = list(zip(texts, images, audios))
    if not triples:
        raise ValueError("input iterables must have equal length")
    return triples


def generate_augmentations(
    triples: Iterable[Tuple[str, torch.Tensor, torch.Tensor]], cfg: IngestConfig
) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Apply simple crops and noise for data augmentation."""
    aug_triples = []
    for text, img, aud in triples:
        h, w = img.shape[-2:]
        if h > cfg.crop_size and w > cfg.crop_size:
            top = random.randint(0, h - cfg.crop_size)
            left = random.randint(0, w - cfg.crop_size)
            img = img[..., top : top + cfg.crop_size, left : left + cfg.crop_size]
        aud = aud + cfg.noise_std * torch.randn_like(aud)
        aug_triples.append((text, img, aud))
    return aug_triples


__all__ = ["IngestConfig", "align_modalities", "generate_augmentations"]
