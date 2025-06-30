from __future__ import annotations

import random
import os
import random
import tarfile
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import requests
import torch
try:  # optional
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore


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


def download_dataset(url: str, dest: str | Path) -> Path:
    """Download and extract an archive if ``dest`` doesn't exist."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    filename = dest / Path(url).name
    if not filename.exists():
        urllib.request.urlretrieve(url, filename)
    if filename.suffix == ".zip":
        with zipfile.ZipFile(filename, "r") as zf:
            zf.extractall(dest)
    elif filename.suffix in {".tar", ".gz", ".tgz"}:
        with tarfile.open(filename) as tf:
            tf.extractall(dest)
    return dest


def _load_image(path: Path) -> torch.Tensor:
    if Image is None:
        raise ImportError("Pillow is required for image loading")
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_audio(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        sampwidth = wf.getsampwidth()
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sampwidth]
        arr = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        arr = arr.reshape(-1, wf.getnchannels()).T
        arr /= float(np.iinfo(dtype).max or 1)
        return torch.from_numpy(arr)


def transcribe_audio(path: str | Path) -> str:
    """Return a transcript using ``speech_recognition`` if available."""
    try:
        import speech_recognition as sr  # pragma: no cover - heavy optional dep
    except Exception:
        return ""
    recog = sr.Recognizer()
    with sr.AudioFile(str(path)) as src:
        audio = recog.record(src)
    try:
        return recog.recognize_google(audio)
    except Exception:
        return ""


@dataclass
class ModalSample:
    text: str
    image: torch.Tensor
    audio: torch.Tensor


def load_pairs(data_dir: str | Path, transcripts: bool = False) -> List[ModalSample]:
    """Load and align ``text``/``image``/``audio`` files under ``data_dir``."""
    base = Path(data_dir)
    texts = {p.stem: p for p in (base / "texts").glob("*.txt")}
    images = {p.stem: p for p in (base / "images").iterdir() if p.is_file()}
    audios = {p.stem: p for p in (base / "audio").iterdir() if p.is_file()}
    keys = texts.keys() & images.keys() & audios.keys()
    items: List[ModalSample] = []
    for k in sorted(keys):
        text = texts[k].read_text().strip()
        if transcripts:
            text = f"{text} {transcribe_audio(audios[k])}".strip()
        img = _load_image(images[k])
        aud = _load_audio(audios[k])
        items.append(ModalSample(text, img, aud))
    return items


def ingest_samples(
    data_dir: str | Path,
    crop_size: Tuple[int, int] | None = (224, 224),
    transcripts: bool = True,
) -> Iterable[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield augmented samples for the fusion and world models."""
    for sample in load_pairs(data_dir, transcripts=transcripts):
        img = random_crop(sample.image, crop_size) if crop_size else sample.image
        yield sample.text, img, sample.audio


def pair_modalities(
    text_dir: str | Path,
    image_dir: str | Path,
    audio_dir: str | Path,
    text_ext: str = ".txt",
    image_ext: str = ".jpg",
    audio_ext: str = ".wav",
) -> List[Tuple[str, str, str]]:
    """Return triples with matching stems across directories."""
    tdir = Path(text_dir)
    idir = Path(image_dir)
    adir = Path(audio_dir)
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


def random_crop(img: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Randomly crop ``img`` (C, H, W) to ``size``."""
    _, h, w = img.shape
    th, tw = size
    if h < th or w < tw:
        raise ValueError("crop size larger than image")
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return img[:, i : i + th, j : j + tw]


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
    "download_dataset",
    "load_pairs",
    "ingest_samples",
    "transcribe_audio",
    "ModalSample",
    "pair_modalities",
    "random_crop_image",
    "add_gaussian_noise",
    "text_dropout",
]
