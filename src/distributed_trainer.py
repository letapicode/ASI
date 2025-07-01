"""Self-healing trainer coordinating checkpoints with DistributedMemory."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Any, Dict

from .distributed_memory import DistributedMemory


@dataclass
class MemoryConfig:
    """Configuration parameters for :class:`DistributedMemory`."""

    dim: int
    compressed_dim: int
    capacity: int
    remotes: list[str] | None = None


def _worker_process(
    train_fn: Callable[[DistributedMemory, int], None],
    mem_cfg: Dict[str, Any],
    ckpt_dir: str,
    step: int,
) -> None:
    """Entry point for each worker process."""
    try:
        prev = Path(ckpt_dir) / f"step{step}"
        if prev.exists():
            mem = DistributedMemory.load(prev / "memory")
            mem.remotes = list(mem_cfg.get("remotes") or [])
        else:
            mem = DistributedMemory(**mem_cfg)
        train_fn(mem, step)
        out = Path(ckpt_dir) / f"step{step + 1}"
        mem.save(out / "memory")
    except Exception:
        # Print traceback for visibility and exit with failure
        import traceback

        traceback.print_exc()
        raise


class DistributedTrainer:
    """Run workers with automatic restarts and checkpointing."""

    def __init__(
        self,
        train_fn: Callable[[DistributedMemory, int], None],
        mem_cfg: MemoryConfig,
        checkpoint_dir: str,
        max_restarts: int = 3,
    ) -> None:
        self.train_fn = train_fn
        self.mem_cfg = mem_cfg.__dict__ if isinstance(mem_cfg, MemoryConfig) else mem_cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_restarts = max_restarts
        self.step = 0

    def run(self, steps: int) -> None:
        """Execute ``train_fn`` for ``steps`` iterations with restart logic."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        restarts = 0
        while self.step < steps:
            proc = mp.Process(
                target=_worker_process,
                args=(self.train_fn, self.mem_cfg, str(self.checkpoint_dir), self.step),
            )
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                restarts += 1
                if restarts > self.max_restarts:
                    raise RuntimeError("Maximum restarts exceeded")
                continue
            restarts = 0
            self.step += 1


__all__ = ["DistributedTrainer", "MemoryConfig"]
