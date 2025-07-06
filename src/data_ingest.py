from __future__ import annotations

import random
import wave
from pathlib import Path
from typing import Iterable, List, Tuple, Callable, Any, Optional, Dict

import numpy as np
import asyncio
import torch
import requests
try:
    import aiohttp  # type: ignore
    _HAS_AIOHTTP = True
except Exception:  # pragma: no cover - optional
    _HAS_AIOHTTP = False
from PIL import Image
try:  # pragma: no cover - allow running without package context
    from .dataset_versioner import DatasetVersioner
except Exception:  # pragma: no cover - for tests
    try:
        from dataset_versioner import DatasetVersioner  # type: ignore
    except Exception:  # pragma: no cover - stub fallback
        class DatasetVersioner:  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def record(self, *args: Any, **kwargs: Any) -> None:
                pass
try:  # pragma: no cover - fallback for local import
    from .generative_data_augmentor import GenerativeDataAugmentor
except Exception:  # pragma: no cover - for tests
    try:
        from generative_data_augmentor import GenerativeDataAugmentor  # type: ignore
    except Exception:  # pragma: no cover - last resort stub
        class GenerativeDataAugmentor:  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def synthesize(self, *args: Any, **kwargs: Any) -> list[tuple[str, np.ndarray, np.ndarray]]:
                return []

try:  # pragma: no cover - optional
    from .dataset_anonymizer import DatasetAnonymizer
except Exception:  # pragma: no cover - for tests
    try:
        from dataset_anonymizer import DatasetAnonymizer  # type: ignore
    except Exception:  # pragma: no cover - stub
        class DatasetAnonymizer:  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def scrub_text_file(self, *a: Any, **kw: Any) -> None:
                pass

            def scrub_image_file(self, *a: Any, **kw: Any) -> None:
                pass

            def scrub_audio_file(self, *a: Any, **kw: Any) -> None:
                pass

            def summary(self) -> Dict[str, int]:
                return {}

try:
    from .dataset_lineage_manager import DatasetLineageManager
except Exception:  # pragma: no cover - for tests
    try:
        from dataset_lineage_manager import DatasetLineageManager  # type: ignore
    except Exception:  # pragma: no cover - stub
        class DatasetLineageManager:  # type: ignore
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def record(self, *a: Any, **kw: Any) -> None:
                pass


try:
    from .data_bias_mitigator import DataBiasMitigator
except Exception:  # pragma: no cover - for tests
    try:
        from data_bias_mitigator import DataBiasMitigator  # type: ignore
    except Exception:
        class DataBiasMitigator:  # type: ignore
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def apply_to_triples(self, triples: Iterable[Tuple[Path, Path, Path]]) -> list[tuple[Path, Path, Path]]:
                return list(triples)

try:
    from .data_poison_detector import DataPoisonDetector
except Exception:  # pragma: no cover - for tests
    try:
        from data_poison_detector import DataPoisonDetector  # type: ignore
    except Exception:
        class DataPoisonDetector:  # type: ignore
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def record_file(self, *a: Any, **kw: Any) -> bool:
                return False

try:  # pragma: no cover - optional bias scoring
    from .dataset_bias_detector import file_bias_score
except Exception:  # pragma: no cover - for tests
    try:
        from dataset_bias_detector import file_bias_score  # type: ignore
    except Exception:  # pragma: no cover - stub
        def file_bias_score(path: str | Path) -> float:  # type: ignore
            return 1.0

try:  # pragma: no cover - optional secure exchange
    from .secure_dataset_exchange import SecureDatasetExchange
except Exception:  # pragma: no cover - for tests
    try:
        from secure_dataset_exchange import SecureDatasetExchange  # type: ignore
    except Exception:  # pragma: no cover - stub
        class SecureDatasetExchange:  # type: ignore
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def push(self, *a: Any, **kw: Any) -> None:
                pass

            def pull(self, *a: Any, **kw: Any) -> None:
                pass



class ActiveDataSelector:
    """Score triples by entropy and output weights in ``[0, 1]``."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def score(self, probs: np.ndarray) -> float:
        """Return entropy of ``probs``."""
        p = probs / (probs.sum() + 1e-8)
        return float(-(p * np.log(p + 1e-8)).sum())

    def select(
        self,
        triples: Iterable[Tuple[Any, Any, Any]],
        probs: Iterable[np.ndarray],
    ) -> list[Tuple[Tuple[Any, Any, Any], float]]:
        """Return ``(triple, weight)`` pairs with bias-adjusted weights."""
        results: list[tuple[tuple[Any, Any, Any], float]] = []
        for t, p in zip(triples, probs):
            ent = self.score(np.asarray(p, dtype=float))
            w = min(ent / (self.threshold + 1e-8), 1.0)
            try:
                bias = max(0.0, float(file_bias_score(t[0])))
                w *= bias
            except Exception:
                pass
            results.append((t, float(w)))
        return results


class CrossLingualTranslator:
    """Translate text to multiple languages using simple placeholders."""

    def __init__(
        self,
        languages: Iterable[str],
        adaptive: "AdaptiveTranslator | None" = None,
    ) -> None:
        self.languages = list(languages)
        self.adaptive = adaptive

    # ------------------------------------------------------
    def _basic_translate(self, text: str, lang: str) -> str:
        if lang not in self.languages:
            raise ValueError(f"unsupported language: {lang}")
        return f"[{lang}] {text}"

    def translate(self, text: str, lang: str | None = None) -> str:
        if lang is None and self.adaptive is not None:
            return self.adaptive.translate(text)
        if lang is None:
            raise ValueError("language must be specified")
        return self._basic_translate(text, lang)

    def translate_all(self, text: str) -> Dict[str, str]:
        return {l: self._basic_translate(text, l) for l in self.languages}


class CrossLingualSpeechTranslator:
    """Offline speech-to-text wrapper using ``speech_recognition``."""

    def __init__(self, translator: CrossLingualTranslator) -> None:
        self.translator = translator
        try:
            import speech_recognition as sr  # type: ignore

            self._sr = sr.Recognizer()
            self._AudioFile = sr.AudioFile
            self._recognize = self._sr.recognize_sphinx
        except Exception:  # pragma: no cover - optional dependency
            self._sr = None
            self._AudioFile = None
            self._recognize = None

    def transcribe(self, audio: str | np.ndarray | torch.Tensor) -> str:
        """Return the transcript of ``audio`` or ``""`` on failure."""
        if self._sr is None or self._AudioFile is None or self._recognize is None:
            return ""

        import tempfile
        import wave

        path: str
        cleanup = False
        if isinstance(audio, str):
            path = audio
        else:
            arr = (
                audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else audio
            )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f.name, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    arr16 = np.asarray(arr, dtype=np.int16)
                    w.writeframes(arr16.tobytes())
                path = f.name
                cleanup = True

        try:
            with self._AudioFile(path) as source:
                data = self._sr.record(source)
            return self._recognize(data)
        except Exception:
            return ""
        finally:
            if cleanup:
                Path(path).unlink(missing_ok=True)

    def translate_all(self, audio: str | np.ndarray | torch.Tensor) -> Dict[str, str]:
        """Return translations for the transcript of ``audio``."""
        txt = self.transcribe(audio)
        return self.translator.translate_all(txt) if txt else {}


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
    versioner: Optional[DatasetVersioner] = None,
    translator: Optional[CrossLingualTranslator] = None,
    anonymizer: Optional[DatasetAnonymizer] = None,
    lineage: Optional[DatasetLineageManager] = None,
    bias_mitigator: Optional["DataBiasMitigator"] = None,
    poison_detector: Optional["DataPoisonDetector"] = None,
) -> List[Tuple[Path, Path, Path]]:
    """Download text, image and audio triples into ``out_dir`` concurrently.

    If ``bias_mitigator`` is provided, downloaded text files are filtered based
    on bias scores before returning the list of triples.
    """

    async def run() -> List[Tuple[Path, Path, Path]]:
        return await download_triples_async(
            text_urls,
            img_urls,
            audio_urls,
            out_dir,
            versioner,
            translator,
            anonymizer,
            lineage,
            bias_mitigator,
            poison_detector,
        )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run())
    else:
        return loop.create_task(run())


async def download_triples_async(
    text_urls: Iterable[str],
    img_urls: Iterable[str],
    audio_urls: Iterable[str],
    out_dir: str,
    versioner: Optional[DatasetVersioner] = None,
    translator: Optional[CrossLingualTranslator] = None,
    anonymizer: Optional[DatasetAnonymizer] = None,
    lineage: Optional[DatasetLineageManager] = None,
    bias_mitigator: Optional["DataBiasMitigator"] = None,
    poison_detector: Optional["DataPoisonDetector"] = None,
) -> List[Tuple[Path, Path, Path]]:
    """Asynchronously download text, image and audio triples.

    The optional ``bias_mitigator`` allows biased samples to be removed before
    they are recorded in the dataset.
    """
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

    if poison_detector is not None:
        filtered: List[Tuple[Path, Path, Path]] = []
        for tri in triples:
            try:
                if not poison_detector.record_file(tri[0]):
                    filtered.append(tri)
            except Exception:
                filtered.append(tri)
        triples = filtered

    if anonymizer is not None:
        for t_path, i_path, a_path in triples:
            try:
                anonymizer.scrub_text_file(t_path)
            except Exception:
                pass
            try:
                anonymizer.scrub_image_file(i_path)
            except Exception:
                pass
            try:
                anonymizer.scrub_audio_file(a_path)
            except Exception:
                pass

    # Add multilingual copies
    if translator is not None:
        augmented: List[Tuple[Path, Path, Path]] = []
        for t_path, i_path, a_path in list(triples):
            augmented.append((t_path, i_path, a_path))
            try:
                txt = t_path.read_text()
            except Exception:
                continue
            for lang, trans in translator.translate_all(txt).items():
                t_new = t_path.with_name(f"{t_path.stem}_{lang}{t_path.suffix}")
                t_new.write_text(trans)
                augmented.append((t_new, i_path, a_path))
        triples = augmented
    if bias_mitigator is not None:
        triples = bias_mitigator.apply_to_triples(triples)
    if lineage is not None and anonymizer is not None:
        flat = [p for tri in triples for p in tri]
        lineage.record(flat, flat, note=f"anonymized {anonymizer.summary()}")
    if versioner is not None:
        flat = [p for tri in triples for p in tri]
        versioner.record(flat, note="download_triples")
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

def synthesize_from_world_model(
    augmentor: GenerativeDataAugmentor,
    seeds: Iterable[Tuple[str, np.ndarray]],
    policy_fn: Callable[[torch.Tensor], torch.Tensor],
    steps: int = 5,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Generate synthetic triples for each ``(text, image)`` seed."""
    triples: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for text, image in seeds:
        triples.extend(augmentor.synthesize(text, image, policy_fn, steps=steps))
    return triples


def offline_synthesizer(
    model: "MultiModalWorldModel",
    tokenizer,
    start_text: str,
    start_image: np.ndarray,
    policy_fn,
    steps: int = 3,
    save_dir: Optional[str | Path] = None,
    versioner: Optional[DatasetVersioner] = None,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Generate synthetic text, image and audio triples via world-model rollout."""

    from asi.multimodal_world_model import rollout  # avoid local import issues
    import torch

    t = torch.tensor(tokenizer(start_text), dtype=torch.long).unsqueeze(0)
    img = torch.tensor(start_image, dtype=torch.float32).unsqueeze(0)

    states, _ = rollout(model, t, img, policy_fn, steps=steps)

    triples: List[Tuple[str, np.ndarray, np.ndarray]] = []
    saved: List[Tuple[Path, Path, Path]] = []
    for idx, s in enumerate(states):
        vec = s.cpu().numpy().ravel()
        txt = " ".join(str(int(x)) for x in vec[:5])
        side = int(np.sqrt(vec.size)) or 1
        img_arr = vec[: side * side].reshape(side, side)
        aud_arr = vec.copy()
        triples.append((txt, img_arr, aud_arr))
        if save_dir is not None:
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            t_p = out / f"syn_{idx}.txt"
            i_p = out / f"syn_{idx}.npy"
            a_p = out / f"syn_{idx}.npy.audio"
            t_p.write_text(txt)
            np.save(i_p, img_arr)
            np.save(a_p, aud_arr)
            saved.append((t_p, i_p, a_p))

    if save_dir is not None and versioner is not None and saved:
        flat = [p for tri in saved for p in tri]
        versioner.record(flat, note="offline_synthesizer")

    return triples


def filter_dataset(text_files: Iterable[str | Path], threshold: float = -3.0) -> List[Path]:
    """Return ``text_files`` filtered by generative noise score."""
    try:
        from .auto_dataset_filter import filter_text_files
    except Exception:  # pragma: no cover - for tests
        import importlib.util
        import sys
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "auto_dataset_filter", Path(__file__).with_name("auto_dataset_filter.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["auto_dataset_filter"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)  # type: ignore
        from auto_dataset_filter import filter_text_files  # type: ignore

    return filter_text_files(text_files, threshold=threshold)


def auto_label_triples(
    triples: Iterable[Tuple[str | Path, str | Path, str | Path]],
    labeler: "AutoLabeler",
) -> list[int]:
    """Apply ``labeler`` to loaded triples and return integer labels."""
    from .auto_labeler import AutoLabeler  # avoid circular import during tests

    if not isinstance(labeler, AutoLabeler):
        raise TypeError("labeler must be AutoLabeler")
    samples = []
    for t_path, i_path, _ in triples:
        text = Path(t_path).read_text()
        if i_path.endswith(".npy"):
            img = np.load(i_path)
        else:
            img = np.array(Image.open(i_path))
        samples.append((text, img, None))
    return labeler.label(samples)


def paraphrase_multilingual(
    text_files: Iterable[str | Path],
    translator: CrossLingualTranslator,
    dataset_filter: Optional[AutoDatasetFilter] = None,
    inspector: Optional["LicenseInspector"] = None,
    lineage: Optional[DatasetLineageManager] = None,
) -> List[Path]:
    """Generate and save paraphrases of ``text_files`` across languages."""

    paths = [Path(p) for p in text_files]
    texts: List[str] = []
    kept_inputs: List[Path] = []
    for p in paths:
        if inspector is not None:
            meta = p.with_suffix(".json")
            if meta.exists() and not inspector.inspect(meta):
                continue
        try:
            txt = p.read_text()
        except Exception:
            continue
        texts.append(txt)
        kept_inputs.append(p)

    if dataset_filter is not None:
        dataset_filter.fit(texts)

    out_paths: List[Path] = []
    total = 0
    for p, txt in zip(kept_inputs, texts):
        for lang, trans in translator.translate_all(f"{txt} paraphrased").items():
            total += 1
            out_p = p.with_name(f"{p.stem}_{lang}_para{p.suffix}")
            if dataset_filter is not None and dataset_filter.score(trans) < dataset_filter.threshold:
                continue
            out_p.write_text(trans)
            out_paths.append(out_p)

    if lineage is not None and out_paths:
        lineage.record(kept_inputs, out_paths, note=f"paraphrase_multilingual generated={total} kept={len(out_paths)}")
    return out_paths


def ingest_translated_triples(
    triples: Iterable[
        Tuple[str | Path, str | Path, str | Path]
        | Tuple[str | Path, str | Path, str | Path, float]
    ],
    tokenizer,
    model: "CrossModalFusion",
    memory: "HierarchicalMemory",
    translator: Optional[CrossLingualTranslator] = None,
    batch_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate ``triples`` and store fused embeddings in ``memory``.

    Each text sample is translated using ``translator`` into all configured
    languages. The resulting text, image, and audio triples are encoded with
    :func:`cross_modal_fusion.encode_all` and the averaged embeddings are stored
    in ``memory`` with a ``{"lang": <code>}`` metadata tag. The function returns
    the raw embeddings for further analysis.
    """

    from .cross_modal_fusion import MultiModalDataset, encode_all

    items: List[Tuple[str, Any, Any]] = []
    metas: List[Dict[str, str]] = []

    for triple in triples:
        if len(triple) == 4:
            t_path, i_path, a_path, ts = triple
        else:
            t_path, i_path, a_path = triple
            ts = None
        text = Path(t_path).read_text()
        if str(i_path).endswith(".npy"):
            image = np.load(i_path)
        else:
            image = np.array(Image.open(i_path))
        if str(a_path).endswith(".npy"):
            audio = np.load(a_path)
        else:
            with wave.open(str(a_path), "rb") as f:
                frames = f.readframes(f.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

        translations = (
            translator.translate_all(text) if translator is not None else {"orig": text}
        )
        for lang, txt in translations.items():
            items.append((txt, image, audio))
            meta = {"lang": lang}
            if ts is not None:
                meta["timestamp"] = float(ts)
            metas.append(meta)

    dataset = MultiModalDataset(items, tokenizer)
    t_vecs, i_vecs, a_vecs = encode_all(model, dataset, batch_size=batch_size)
    memory.add_multimodal(t_vecs, i_vecs, a_vecs, metas)
    return t_vecs, i_vecs, a_vecs


def push_dataset(
    directory: str | Path,
    package: str | Path,
    key: bytes,
    signing_key: bytes | None = None,
) -> Path:
    """Encrypt ``directory`` and store it as ``package``."""
    exchange = SecureDatasetExchange(key, signing_key=signing_key)
    return exchange.push(directory, package)


def pull_dataset(
    package: str | Path,
    directory: str | Path,
    key: bytes,
    verify_key: bytes | None = None,
) -> None:
    """Decrypt ``package`` into ``directory``."""
    exchange = SecureDatasetExchange(key, verify_key=verify_key)
    exchange.pull(package, directory)


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
    "synthesize_from_world_model",
    "offline_synthesizer",
    "filter_dataset",
    "auto_label_triples",
    "paraphrase_multilingual",
    "ingest_translated_triples",
    "push_dataset",
    "pull_dataset",
    "ActiveDataSelector",
    "CrossLingualTranslator",
    "CrossLingualSpeechTranslator",
    "DataPoisonDetector",
]
