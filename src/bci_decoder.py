import numpy as np

try:  # pragma: no cover - optional dependency
    import mne  # type: ignore
    _HAS_MNE = True
except Exception:  # pragma: no cover - during tests
    _HAS_MNE = False


def load_bci(path: str) -> np.ndarray:
    """Return EEG/ECoG signal data loaded via ``mne``."""
    if not _HAS_MNE:
        raise ImportError("mne not available")
    raw = mne.io.read_raw(path, preload=True)
    return raw.get_data().astype("float32")


__all__ = ["load_bci", "_HAS_MNE"]
