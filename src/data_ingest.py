import os
import random
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore


def download_dataset(url: str, dest: str | Path) -> Path:
    """Download and extract an archive if ``dest`` doesn't already exist."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    filename = dest / Path(url).name
    if not filename.exists():
        urllib.request.urlretrieve(url, filename)
    if filename.suffix in {".zip"}:
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
    import wave

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


def random_crop(img: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Randomly crop ``img`` (C, H, W) to ``size``."""
    _, h, w = img.shape
    th, tw = size
    if h < th or w < tw:
        raise ValueError("crop size larger than image")
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return img[:, i : i + th, j : j + tw]


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
    """Yield augmented samples suitable for the fusion and world models."""
    for sample in load_pairs(data_dir, transcripts=transcripts):
        img = random_crop(sample.image, crop_size) if crop_size else sample.image
        yield sample.text, img, sample.audio


__all__ = [
    "download_dataset",
    "load_pairs",
    "ingest_samples",
    "transcribe_audio",
    "random_crop",
    "ModalSample",
]

