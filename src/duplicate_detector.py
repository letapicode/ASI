import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

try:  # optional heavy dependency
    import open_clip  # type: ignore
    _HAS_OPENCLIP = True
except Exception:  # pragma: no cover - optional
    _HAS_OPENCLIP = False

import torch


class _LSHIndex:
    """Simple locality-sensitive hash index using random projections."""

    def __init__(self, dim: int, num_planes: int = 16) -> None:
        self.dim = dim
        self.num_planes = num_planes
        rng = np.random.RandomState(0)
        self.planes = rng.randn(num_planes, dim).astype(np.float32)
        self.buckets: Dict[Tuple[int, ...], List[np.ndarray]] = {}

    def _hash(self, vec: np.ndarray) -> Tuple[int, ...]:
        signs = (vec @ self.planes.T) >= 0
        return tuple(int(s) for s in signs)

    def add(self, vec: np.ndarray) -> None:
        key = self._hash(vec)
        self.buckets.setdefault(key, []).append(vec)

    def query(self, vec: np.ndarray) -> Iterable[np.ndarray]:
        key = self._hash(vec)
        return self.buckets.get(key, [])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(a.dot(b) / (an * bn))


class DuplicateDetector:
    """Detect near-duplicate texts or images using CLIP embeddings and LSH."""

    def __init__(self, threshold: float = 0.95, device: str | None = None) -> None:
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.index = _LSHIndex(512)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        if _HAS_OPENCLIP:
            # use a lightweight pretrained model
            self.model, self.preprocess, self.tokenizer = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self.model.to(self.device)
            self.model.eval()

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _embed_text(self, text: str) -> np.ndarray:
        if self.model is not None and self.tokenizer is not None:
            tok = self.tokenizer([text])
            with torch.no_grad():
                out = self.model.encode_text(tok.to(self.device))
            return out[0].cpu().numpy().astype(np.float32)
        # fallback: simple hash embedding
        h = hashlib.sha1(text.encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        if arr.size < 512:
            arr = np.pad(arr, (0, 512 - arr.size))
        else:
            arr = arr[:512]
        return arr

    def _embed_image(self, path: str | Path) -> np.ndarray:
        if self.model is not None and self.preprocess is not None:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model.encode_image(tensor)
            return out[0].cpu().numpy().astype(np.float32)
        # fallback: use file bytes hash
        data = Path(path).read_bytes()
        h = hashlib.sha1(data).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        if arr.size < 512:
            arr = np.pad(arr, (0, 512 - arr.size))
        else:
            arr = arr[:512]
        return arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_duplicate_text(self, text: str) -> bool:
        vec = self._embed_text(text)
        for v in self.index.query(vec):
            if _cosine_similarity(vec, v) >= self.threshold:
                return True
        self.index.add(vec)
        return False

    def is_duplicate_image(self, path: str | Path) -> bool:
        vec = self._embed_image(path)
        for v in self.index.query(vec):
            if _cosine_similarity(vec, v) >= self.threshold:
                return True
        self.index.add(vec)
        return False

    def filter_texts(self, texts: Iterable[str]) -> List[str]:
        kept = []
        for t in texts:
            if not self.is_duplicate_text(t):
                kept.append(t)
        return kept

    def filter_files(self, files: Iterable[str | Path]) -> List[Path]:
        kept = []
        for f in files:
            if not self.is_duplicate_image(f):
                kept.append(Path(f))
        return kept


__all__ = ["DuplicateDetector"]
