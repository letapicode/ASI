"""Offline memory replay utilities."""

from __future__ import annotations

from typing import Any

import torch

from .hierarchical_memory import HierarchicalMemory
from .context_summary_memory import ContextSummaryMemory
from .distributed_trainer import DistributedTrainer


class BioMemoryReplayer:
    """Reconstruct stored embeddings and run them through a model."""

    def __init__(self, model: torch.nn.Module, memory: HierarchicalMemory | ContextSummaryMemory, batch_size: int = 32) -> None:
        self.model = model
        self.memory = memory
        self.batch_size = batch_size

    def reconstruct_sequences(self) -> tuple[torch.Tensor, list[Any]]:
        """Return reconstructed embeddings and their metadata."""
        data = []
        out_meta: list[Any] = []
        metas = getattr(self.memory.store, "_meta", [])
        for vec, meta in zip(self.memory.compressor.buffer.data, metas):
            if (
                isinstance(self.memory, ContextSummaryMemory)
                and isinstance(meta, dict)
                and "ctxsum" in meta
            ):
                text = meta["ctxsum"]["summary"]
                vec = self.memory.summarizer.expand(text)
            data.append(vec.detach().clone())
            out_meta.append(meta)
        if data:
            return torch.stack(data), out_meta
        dim = self.memory.compressor.encoder.in_features
        return torch.empty(0, dim), []

    def replay(self) -> None:
        """Feed reconstructed embeddings through the model."""
        seqs, metas = self.reconstruct_sequences()
        if seqs.numel() == 0:
            return
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                batch = seqs[i : i + self.batch_size].to(device)
                out = self.model(batch)
                comp = self.memory.compressor.encoder(out).cpu()
                self.memory.add_compressed(comp, metas[i : i + self.batch_size])


def run_nightly_replay(trainer: DistributedTrainer, model: torch.nn.Module, memory: HierarchicalMemory | ContextSummaryMemory, batch_size: int = 32) -> None:
    """Attach nightly memory replay to a trainer."""
    replayer = BioMemoryReplayer(model, memory, batch_size)
    trainer.replay_hook = replayer.replay
    trainer.replay_interval = 24 * 3600


__all__ = ["BioMemoryReplayer", "run_nightly_replay"]
