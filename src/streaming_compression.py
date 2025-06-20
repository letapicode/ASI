import random
from typing import List

import torch
from torch import nn


class ReservoirBuffer:
    """Maintain a reservoir sample of token embeddings."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: List[torch.Tensor] = []
        self.count = 0

    def add(self, x: torch.Tensor) -> None:
        """Add a batch of embeddings to the reservoir."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for row in x:
            self.count += 1
            if len(self.data) < self.capacity:
                self.data.append(row.detach().clone())
            else:
                j = random.randint(0, self.count - 1)
                if j < self.capacity:
                    self.data[j] = row.detach().clone()

    def get(self) -> torch.Tensor:
        if not self.data:
            return torch.empty(0)
        return torch.stack(self.data)


class StreamingCompressor(nn.Module):
    """Reservoir sampling with a small autoencoder for lossy storage."""

    def __init__(self, dim: int, compressed_dim: int, capacity: int) -> None:
        super().__init__()
        self.buffer = ReservoirBuffer(capacity)
        self.encoder = nn.Linear(dim, compressed_dim)
        self.decoder = nn.Linear(compressed_dim, dim)

    def add(self, x: torch.Tensor) -> None:
        """Store new embeddings in the reservoir."""
        self.buffer.add(x)

    def compressed(self) -> torch.Tensor:
        """Return compressed representations for the buffered data."""
        data = self.buffer.get()
        if data.numel() == 0:
            return data
        return self.encoder(data)

    def reconstruct(self) -> torch.Tensor:
        """Decode the compressed reservoir back to the original dimension."""
        comp = self.compressed()
        if comp.numel() == 0:
            return comp
        return self.decoder(comp)
