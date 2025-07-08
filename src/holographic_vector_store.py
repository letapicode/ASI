import numpy as np
from pathlib import Path
from typing import Iterable, Any, List, Tuple

from .vector_store import VectorStore


def _convolve_freq(a_f: np.ndarray, b_f: np.ndarray, n: int) -> np.ndarray:
    out = np.fft.irfft(a_f * b_f, n=n)
    return out.astype(np.float32)


def _correlate_freq(a_f: np.ndarray, b_f: np.ndarray, n: int) -> np.ndarray:
    out = np.fft.irfft(a_f * np.conj(b_f), n=n)
    return out.astype(np.float32)


class HolographicVectorStore:
    """Vector store using holographic reduced representations."""

    def __init__(self, dim: int, path: str | Path | None = None) -> None:
        self.dim = dim
        self.store = VectorStore(dim)
        self.keys = {
            "text": np.random.randn(dim).astype(np.float32),
            "image": np.random.randn(dim).astype(np.float32),
            "audio": np.random.randn(dim).astype(np.float32),
        }
        self._fkeys = {k: np.fft.rfft(v) for k, v in self.keys.items()}
        self.path = Path(path) if path else None
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            key_file = self.path / "hrr_keys.npz"
            store_file = self.path / "store.npz"
            if key_file.exists():
                data = np.load(key_file)
                for k in self.keys:
                    if k in data:
                        self.keys[k] = data[k]
                self._fkeys = {k: np.fft.rfft(self.keys[k]) for k in self.keys}
            if store_file.exists():
                self.store = VectorStore.load(store_file)

    # --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.store)

    # --------------------------------------------------------------
    def encode(
        self,
        text: np.ndarray | None = None,
        image: np.ndarray | None = None,
        audio: np.ndarray | None = None,
    ) -> np.ndarray:
        out_f = np.zeros_like(self._fkeys["text"], dtype=np.complex64)
        if text is not None:
            out_f += np.fft.rfft(text.astype(np.float32)) * self._fkeys["text"]
        if image is not None:
            out_f += np.fft.rfft(image.astype(np.float32)) * self._fkeys["image"]
        if audio is not None:
            out_f += np.fft.rfft(audio.astype(np.float32)) * self._fkeys["audio"]
        out = np.fft.irfft(out_f, n=self.dim)
        return out.astype(np.float32)

    # --------------------------------------------------------------
    def encode_batch(
        self,
        text: np.ndarray | None = None,
        image: np.ndarray | None = None,
        audio: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode multiple modality arrays at once."""
        if text is not None:
            n = text.shape[0]
        elif image is not None:
            n = image.shape[0]
        elif audio is not None:
            n = audio.shape[0]
        else:
            return np.empty((0, self.dim), dtype=np.float32)
        out_f = np.zeros((n, self._fkeys["text"].shape[0]), dtype=np.complex64)
        if text is not None:
            out_f += np.fft.rfft(text.astype(np.float32), axis=1) * self._fkeys["text"]
        if image is not None:
            out_f += np.fft.rfft(image.astype(np.float32), axis=1) * self._fkeys["image"]
        if audio is not None:
            out_f += np.fft.rfft(audio.astype(np.float32), axis=1) * self._fkeys["audio"]
        out = np.fft.irfft(out_f, n=self.dim, axis=1)
        return out.astype(np.float32)

    # --------------------------------------------------------------
    def decode(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vf = np.fft.rfft(vec.astype(np.float32))
        text = _correlate_freq(vf, self._fkeys["text"], self.dim)
        image = _correlate_freq(vf, self._fkeys["image"], self.dim)
        audio = _correlate_freq(vf, self._fkeys["audio"], self.dim)
        return text, image, audio

    # --------------------------------------------------------------
    def add(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        self.store.add(vectors, metadata)

    # --------------------------------------------------------------
    def add_modalities(
        self,
        text: np.ndarray,
        image: np.ndarray,
        audio: np.ndarray,
        metadata: Iterable[Any] | None = None,
    ) -> None:
        arr = self.encode_batch(text, image, audio)
        self.add(arr, metadata)

    # --------------------------------------------------------------
    def delete(self, index: int | Iterable[int] | None = None, tag: Any | None = None) -> None:
        self.store.delete(index=index, tag=tag)

    # --------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        return self.store.search(query, k)

    # --------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.store.save(path / "store.npz")
        np.savez(path / "hrr_keys.npz", **self.keys)

    # --------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "HolographicVectorStore":
        path = Path(path)
        key_file = path / "hrr_keys.npz"
        store_file = path / "store.npz"
        if not store_file.exists():
            raise FileNotFoundError(store_file)
        store = cls(int(np.load(key_file)["text"].shape[0]), path)
        store.store = VectorStore.load(store_file)
        return store


__all__ = ["HolographicVectorStore"]
