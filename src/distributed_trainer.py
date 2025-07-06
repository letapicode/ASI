"""Self-healing trainer coordinating checkpoints with DistributedMemory."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Any, Dict
import time
import sys
import json

import torch

from .gradient_compression import GradientCompressionConfig, GradientCompressor
from .telemetry import TelemetryLogger
from .adaptive_micro_batcher import AdaptiveMicroBatcher
from .gpu_aware_scheduler import GPUAwareScheduler
from .hpc_scheduler import submit_job
from .enclave_runner import EnclaveRunner, EnclaveConfig


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
    enclave_cfg: Dict[str, Any] | None,
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
        runner = EnclaveRunner(EnclaveConfig(**enclave_cfg)) if enclave_cfg else None
        if runner is not None:
            runner.run(train_fn, mem, step, fn)
        else:
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
        scheduler: GPUAwareScheduler | None = None,
        micro_batcher: AdaptiveMicroBatcher | None = None,
        hpc_backend: str | None = None,
        enclave: EnclaveConfig | None = None,

    ) -> None:
        self.train_fn = train_fn
        self.train_fn_path = f"{train_fn.__module__}:{train_fn.__name__}"
        self.mem_cfg = mem_cfg.__dict__ if isinstance(mem_cfg, MemoryConfig) else mem_cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_restarts = max_restarts
        self.step = 0
        self.grad_compression = grad_compression.__dict__ if isinstance(grad_compression, GradientCompressionConfig) else grad_compression
        self.telemetry = telemetry
        self.scheduler = scheduler
        self.micro_batcher = micro_batcher
        self.hpc_backend = hpc_backend
        self.enclave_cfg = enclave.__dict__ if isinstance(enclave, EnclaveConfig) else enclave


    def run(self, steps: int) -> None:
        """Execute ``train_fn`` for ``steps`` iterations with restart logic."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        restarts = 0
        if self.telemetry:
            self.telemetry.start()
        if self.micro_batcher:
            self.micro_batcher.start()
        while self.step < steps:
            if self.hpc_backend:
                cmd = [
                    sys.executable,
                    __file__,
                    "--train-fn",
                    self.train_fn_path,
                    "--mem-cfg",
                    json.dumps(self.mem_cfg),
                    "--ckpt-dir",
                    str(self.checkpoint_dir),
                    "--step",
                    str(self.step),
                ]
                if self.grad_compression is not None:
                    cmd += ["--comp-cfg", json.dumps(self.grad_compression)]
                if self.scheduler is not None and hasattr(self.scheduler, "submit_job"):
                    self.scheduler.submit_job(cmd, backend=self.hpc_backend)
                else:
                    submit_job(cmd, backend=self.hpc_backend)
                self.step += 1
                continue

            proc = mp.Process(
                target=_worker_process,
                args=(
                    self.train_fn,
                    self.mem_cfg,
                    str(self.checkpoint_dir),
                    self.step,
                    self.grad_compression,
                    self.enclave_cfg,
                ),
            )
            if self.scheduler is not None:
                started = mp.Event()

                def _start() -> None:
                    proc.start()
                    started.set()

                self.scheduler.add(_start)
                while not started.is_set():
                    time.sleep(self.scheduler.check_interval)
            else:
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
            if self.micro_batcher:
                self.micro_batcher.tick()
        if self.telemetry:
            self.telemetry.stop()
        if self.micro_batcher:
            self.micro_batcher.stop()


__all__ = ["DistributedTrainer", "MemoryConfig", "GradientCompressionConfig"]


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse
    parser = argparse.ArgumentParser(description="Run worker process")
    parser.add_argument("--train-fn", required=True)
    parser.add_argument("--mem-cfg", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--comp-cfg", default=None)
    args = parser.parse_args()

    mod_name, fn_name = args.train_fn.rsplit(":", 1)
    module = __import__(mod_name, fromlist=[fn_name])
    fn = getattr(module, fn_name)
    mem_cfg = json.loads(args.mem_cfg)
    comp = json.loads(args.comp_cfg) if args.comp_cfg else None
    _worker_process(fn, mem_cfg, args.ckpt_dir, args.step, comp)

__all__ = [
    "DistributedTrainer",
    "MemoryConfig",
    "GradientCompressionConfig",
    "EnclaveConfig",
]

