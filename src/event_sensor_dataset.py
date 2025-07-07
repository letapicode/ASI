import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Iterable, Sequence, Union


class EventSensorDataset(Dataset):
    """Dataset for neuromorphic event streams."""

    def __init__(
        self,
        streams: Sequence[Union[np.ndarray, str]],
        normalize: bool = True,
    ) -> None:
        self.data = []
        for s in streams:
            if isinstance(s, str):
                arr = np.load(s).astype("float32")
            else:
                arr = np.asarray(s, dtype="float32")
            tensor = torch.tensor(arr, dtype=torch.float32)
            if normalize:
                tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
            self.data.append(tensor)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def load_synthetic_events(num_samples: int = 4, channels: int = 2, length: int = 32) -> EventSensorDataset:
    """Return random event streams for testing."""
    rng = np.random.default_rng(0)
    streams = [rng.standard_normal((channels, length)).astype("float32") for _ in range(num_samples)]
    return EventSensorDataset(streams)


__all__ = ["EventSensorDataset", "load_synthetic_events"]
