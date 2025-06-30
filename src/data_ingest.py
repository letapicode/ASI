from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import random


def pair_modalities(
    text_dir: str | Path,
    image_dir: str | Path,
    audio_dir: str | Path,
    text_ext: str = ".txt",
    image_ext: str = ".jpg",
    audio_ext: str = ".wav",
) -> List[Tuple[str, str, str]]:
    """Return triples of file paths with matching stems across the three directories."""
    tdir = Path(text_dir)
    idir = Path(image_dir)
    adir = Path(audio_dir)
    stems = {p.stem for p in tdir.glob(f"*{text_ext}")}
    stems &= {p.stem for p in idir.glob(f"*{image_ext}")}
    stems &= {p.stem for p in adir.glob(f"*{audio_ext}")}
    pairs = [
        (
            str(tdir / f"{s}{text_ext}"),
            str(idir / f"{s}{image_ext}"),
            str(adir / f"{s}{audio_ext}"),
        )
        for s in sorted(stems)
    ]
    return pairs


def random_crop_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Return a random crop of ``image`` with (height, width) ``size``."""
    h, w = image.shape[:2]
    th, tw = size
    if th > h or tw > w:
        raise ValueError("crop size exceeds image dimensions")
    top = random.randint(0, h - th)
    left = random.randint(0, w - tw)
    return image[top : top + th, left : left + tw]


def add_gaussian_noise(audio: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise with standard deviation ``std`` to ``audio``."""
    noise = np.random.normal(0.0, std, size=audio.shape)
    return (audio + noise).astype(audio.dtype)


def text_dropout(text: str, p: float = 0.1) -> str:
    """Randomly drop words from ``text`` with probability ``p``."""
    words = text.split()
    kept = [w for w in words if random.random() > p]
    if not kept and words:
        kept.append(words[0])
    return " ".join(kept)


__all__ = [
    "pair_modalities",
    "random_crop_image",
    "add_gaussian_noise",
    "text_dropout",
]
