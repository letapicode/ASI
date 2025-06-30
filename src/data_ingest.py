from __future__ import annotations

import random
import wave
from pathlib import Path
from typing import Iterable, Tuple, List

import requests
from PIL import Image


def download_file(url: str, dest: Path) -> None:
    """Download ``url`` into ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)


def download_triples(
    text_urls: Iterable[str],
    img_urls: Iterable[str],
    audio_urls: Iterable[str],
    out_dir: str,
) -> List[Tuple[Path, Path, Path]]:
    """Download text, image and audio triples into ``out_dir``."""
    triples: List[Tuple[Path, Path, Path]] = []
    out = Path(out_dir)
    for i, (t, iurl, a) in enumerate(zip(text_urls, img_urls, audio_urls)):
        t_path = out / "text" / f"{i}.txt"
        i_path = out / "images" / f"{i}.png"
        a_path = out / "audio" / f"{i}.wav"
        download_file(t, t_path)
        download_file(iurl, i_path)
        download_file(a, a_path)
        triples.append((t_path, i_path, a_path))
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


__all__ = [
    "download_triples",
    "align_triples",
    "random_crop",
    "generate_transcript",
]
