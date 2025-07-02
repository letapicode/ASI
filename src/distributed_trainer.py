"""Self-healing trainer coordinating checkpoints with DistributedMemory."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Any, Dict

import torch

from .gradient_compression import GradientCompressionConfig, GradientCompressor
from .telemetry import TelemetryLogger

from .distributed_memory import DistributedMemory


@dataclass
class MemoryConfig:
    """Configuration parameters for :class:`DistributedMemory`."""

    dim: int
    compressed_dim: int
    capacity: int
    remotes: list[str] | None = None


def _worker_process(
    train_fn: Callable[[DistributedMemory, int, Callable[[torch.Tensor], torch.Tensor] | None], None],
    mem_cfg: Dict[str, Any],
    ckpt_dir: str,
    step: int,
    comp_cfg: Dict[str, Any] | None,
) -> None:
    """Entry point for each worker process."""
    try:
        prev = Path(ckpt_dir) / f"step{step}"
        if prev.exists():
            mem = DistributedMemory.load(prev / "memory")
            mem.remotes = list(mem_cfg.get("remotes") or [])
        else:
            mem = DistributedMemory(**mem_cfg)
        compressor = None
        if comp_cfg is not None:
            cfg = GradientCompressionConfig(**comp_cfg)
            compressor = GradientCompressor(cfg)
        fn = compressor.compress if compressor is not None else None
        train_fn(mem, step, fn)
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
        train_fn: Callable[[DistributedMemory, int, Callable[[torch.Tensor], torch.Tensor] | None], None],
        mem_cfg: MemoryConfig,
        checkpoint_dir: str,
        max_restarts: int = 3,
        grad_compression: GradientCompressionConfig | None = None,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self.train_fn = train_fn
        self.mem_cfg = mem_cfg.__dict__ if isinstance(mem_cfg, MemoryConfig) else mem_cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_restarts = max_restarts
        self.step = 0
        self.grad_compression = grad_compression.__dict__ if isinstance(grad_compression, GradientCompressionConfig) else grad_compression
        self.telemetry = telemetry

    def run(self, steps: int) -> None:
        """Execute ``train_fn`` for ``steps`` iterations with restart logic."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        restarts = 0
        if self.telemetry:
            self.telemetry.start()
        while self.step < steps:
            proc = mp.Process(
                target=_worker_process,
                args=(
                    self.train_fn,
                    self.mem_cfg,
                    str(self.checkpoint_dir),
                    self.step,
                    self.grad_compression,
                ),
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
            if self.telemetry:
                stats = self.telemetry.get_stats()
                stats["step"] = self.step
                print("telemetry", stats)
        if self.telemetry:
            self.telemetry.stop()


__all__ = ["DistributedTrainer", "MemoryConfig", "GradientCompressionConfig"]
