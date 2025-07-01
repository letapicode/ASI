from __future__ import annotations

from __future__ import annotations

import random
import wave
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import asyncio
import requests
try:
    import aiohttp  # type: ignore
    _HAS_AIOHTTP = True
except Exception:  # pragma: no cover - optional
    _HAS_AIOHTTP = False
from PIL import Image


def download_file(url: str, dest: Path) -> None:
    """Download ``url`` into ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)


async def _download_file_async(session: aiohttp.ClientSession, url: str, dest: Path) -> None:
    """Asynchronously download ``url`` to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with session.get(url, timeout=30) as resp:
        resp.raise_for_status()
        dest.write_bytes(await resp.read())


def download_triples(
    text_urls: Iterable[str],
    img_urls: Iterable[str],
    audio_urls: Iterable[str],
    out_dir: str,
) -> List[Tuple[Path, Path, Path]]:
    """Download text, image and audio triples into ``out_dir``."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(download_triples_async(text_urls, img_urls, audio_urls, out_dir))
    else:
        return loop.create_task(download_triples_async(text_urls, img_urls, audio_urls, out_dir))


async def download_triples_async(
    text_urls: Iterable[str],
    img_urls: Iterable[str],
    audio_urls: Iterable[str],
    out_dir: str,
) -> List[Tuple[Path, Path, Path]]:
    """Asynchronously download text, image and audio triples."""
    if not _HAS_AIOHTTP:
        raise ImportError("aiohttp is required for async downloads")
    out = Path(out_dir)
    triples: List[Tuple[Path, Path, Path]] = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, (t, iurl, a) in enumerate(zip(text_urls, img_urls, audio_urls)):
            t_path = out / "text" / f"{i}.txt"
            i_path = out / "images" / f"{i}.png"
            a_path = out / "audio" / f"{i}.wav"
            triples.append((t_path, i_path, a_path))
            tasks.append(_download_file_async(session, t, t_path))
            tasks.append(_download_file_async(session, iurl, i_path))
            tasks.append(_download_file_async(session, a, a_path))
        await asyncio.gather(*tasks)
    return triples


def align_triples(text_dir: str, img_dir: str, audio_dir: str) -> List[Tuple[Path, Path, Path]]:
    """Align triples by matching basenames across three directories."""
    texts = {p.stem: p for p in Path(text_dir).glob("*.txt")}
    imgs = {p.stem: p for p in Path(img_dir).glob("*.png")}
    auds = {p.stem: p for p in Path(audio_dir).glob("*.wav")}
    keys = texts.keys() & imgs.keys() & auds.keys()
    return [(texts[k], imgs[k], auds[k]) for k in sorted(keys)]


def random_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Return a random crop of ``img`` with ``size``."""
    w, h = img.size
    cw, ch = size
    if cw > w or ch > h:
        raise ValueError("crop size larger than image")
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    return img.crop((x, y, x + cw, y + ch))


def generate_transcript(audio_path: str | Path) -> str:
    """Return a dummy transcript describing the audio duration."""
    with wave.open(str(audio_path)) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return f"duration:{duration:.2f}s"


def pair_modalities(
    text_dir: str | Path,
    image_dir: str | Path,
    audio_dir: str | Path,
    text_ext: str = ".txt",
    image_ext: str = ".jpg",
    audio_ext: str = ".wav",
) -> List[Tuple[str, str, str]]:
    """Return triples of file paths with matching stems in the three folders."""
    tdir, idir, adir = Path(text_dir), Path(image_dir), Path(audio_dir)
    stems = {p.stem for p in tdir.glob(f"*{text_ext}")}
    stems &= {p.stem for p in idir.glob(f"*{image_ext}")}
    stems &= {p.stem for p in adir.glob(f"*{audio_ext}")}
    return [
        (
            str(tdir / f"{s}{text_ext}"),
            str(idir / f"{s}{image_ext}"),
            str(adir / f"{s}{audio_ext}"),
        )
        for s in sorted(stems)
    ]


def random_crop_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Return a random crop of ``image`` with ``(h, w)`` ``size``."""
    h, w = image.shape[:2]
    th, tw = size
    if th > h or tw > w:
        raise ValueError("crop size exceeds image dimensions")
    top = random.randint(0, h - th)
    left = random.randint(0, w - tw)
    return image[top : top + th, left : left + tw]


def add_gaussian_noise(audio: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Return ``audio`` with Gaussian noise of standard deviation ``std``."""
    noise = np.random.normal(0.0, std, size=audio.shape)
    return (audio + noise).astype(audio.dtype)


def text_dropout(text: str, p: float = 0.1) -> str:
    """Randomly drop words from ``text`` with probability ``p``."""
    words = text.split()
    kept = [w for w in words if random.random() > p]
    if not kept and words:
        kept.append(words[0])
    return " ".join(kept)


def offline_synthesizer(
    model: "MultiModalWorldModel",
    tokenizer,
    start_text: str,
    start_image: np.ndarray,
    policy_fn,
    steps: int = 3,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Generate synthetic text, image and audio triples via world-model rollout.

    The function performs a short rollout of ``model`` starting from
    ``start_text`` and ``start_image``. Each predicted latent state is
    deterministically mapped back to toy text, image and audio
    representations so that downstream modules can consume the data without
    requiring a heavy decoder.

    Parameters
    ----------
    model:
        Trained :class:`~asi.multimodal_world_model.MultiModalWorldModel`.
    tokenizer:
        Callable used to tokenize text inputs for the world model.
    start_text:
        Initial text prompt.
    start_image:
        Initial image array.
    policy_fn:
        Function mapping latent states to actions.
    steps:
        Number of rollout steps to generate.

    Returns
    -------
    list[tuple[str, np.ndarray, np.ndarray]]
        Synthetic ``(text, image, audio)`` triples.
    """

    from asi.multimodal_world_model import rollout  # avoid local import issues
    import torch

    t = torch.tensor(tokenizer(start_text), dtype=torch.long).unsqueeze(0)
    img = torch.tensor(start_image, dtype=torch.float32).unsqueeze(0)

    states, _ = rollout(model, t, img, policy_fn, steps=steps)

    triples: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for s in states:
        vec = s.cpu().numpy().ravel()
        txt = " ".join(str(int(x)) for x in vec[:5])
        side = int(np.sqrt(vec.size)) or 1
        img_arr = vec[: side * side].reshape(side, side)
        aud_arr = vec.copy()
        triples.append((txt, img_arr, aud_arr))

    return triples


__all__ = [
    "download_triples",
    "download_triples_async",
    "align_triples",
    "random_crop",
    "generate_transcript",
    "pair_modalities",
    "random_crop_image",
    "add_gaussian_noise",
    "text_dropout",
    "offline_synthesizer",
]
