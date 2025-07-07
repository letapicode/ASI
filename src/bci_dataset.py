from typing import Iterable
import numpy as np
import torch
from torch.utils.data import Dataset


class BCIDataset(Dataset):
    """Simple dataset for BCI recordings."""

    def __init__(self, recordings: Iterable[np.ndarray], normalize: bool = True) -> None:
        self.data = []
        for rec in recordings:
            arr = torch.tensor(rec, dtype=torch.float32)
            if normalize:
                arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            self.data.append(arr)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def load_synthetic_bci(num_samples: int = 4, channels: int = 2, length: int = 32) -> BCIDataset:
    """Return a :class:`BCIDataset` with random signals."""
    rng = np.random.default_rng(0)
    recs = [rng.standard_normal((channels, length)).astype("float32") for _ in range(num_samples)]
    return BCIDataset(recs)


__all__ = ["BCIDataset", "load_synthetic_bci"]
