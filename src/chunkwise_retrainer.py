import torch
from torch import nn
from typing import Iterable

class ChunkWiseRetrainer:
    """Retrain a model on token streams in fixed-size chunks."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, chunk_size: int) -> None:
        self.model = model
        self.optimizer = optimizer
        self.chunk_size = chunk_size
        self.criterion = nn.CrossEntropyLoss()

    def train(self, sequences: Iterable[torch.Tensor], epochs: int = 1) -> float:
        """Train ``model`` on the given sequences.

        The sequences are split into chunks of ``chunk_size`` tokens.
        Each chunk uses next-token prediction with cross-entropy loss.
        Returns the average loss over all chunks.
        """
        total_loss = 0.0
        count = 0
        for seq in sequences:
            for start in range(0, seq.size(0) - self.chunk_size, self.chunk_size):
                inp = seq[start : start + self.chunk_size]
                tgt = seq[start + 1 : start + 1 + self.chunk_size]
                for _ in range(epochs):
                    logits = self.model(inp.unsqueeze(0))
                    loss = self.criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                count += 1
        return total_loss / max(count, 1)
