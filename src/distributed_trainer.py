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
from .accelerator_scheduler import AcceleratorScheduler
from .hpc_schedulers import submit_job
from .enclave_runner import EnclaveRunner, EnclaveConfig


from .distributed_memory import DistributedMemory
import numpy as np


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
    worker_id: int = 0,
    steps: int = 1,
) -> None:
    """Run ``train_fn`` for ``steps`` iterations and save checkpoints."""
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
        for s in range(step, step + steps):
            if runner is not None:
                runner.run(train_fn, mem, s, fn)
            else:
                train_fn(mem, s, fn)
            out = Path(ckpt_dir) / f"step{s + 1}"
            if worker_id:
                out = out.with_name(out.name + f"_w{worker_id}")
            mem.save(out / "memory")
    except Exception:
        # Print traceback for visibility and exit with failure
        import traceback

        traceback.print_exc()
        raise


def _average_memories(mems: list[DistributedMemory]) -> DistributedMemory:
    """Average compressor parameters and merge buffers."""
    if len(mems) == 1:
        return mems[0]
    base = mems[0]
    n = float(len(mems))
    # average encoder and decoder weights
    enc_keys = base.compressor.encoder.state_dict().keys()
    enc_avg = {}
    for k in enc_keys:
        enc_avg[k] = sum(m.compressor.encoder.state_dict()[k] for m in mems) / n
    dec_keys = base.compressor.decoder.state_dict().keys()
    dec_avg = {}
    for k in dec_keys:
        dec_avg[k] = sum(m.compressor.decoder.state_dict()[k] for m in mems) / n
    base.compressor.encoder.load_state_dict(enc_avg)
    base.compressor.decoder.load_state_dict(dec_avg)
    # merge buffers
    data = []
    for m in mems:
        data.extend([t.clone() for t in m.compressor.buffer.data])
    cap = base.compressor.buffer.capacity
    data = data[-cap:]
    base.compressor.buffer.data = data
    base.compressor.buffer.count = len(data)
    # merge vector stores when possible
    try:
        from .vector_store import VectorStore

        if isinstance(base.store, VectorStore):
            vecs = []
            metas = []
            for m in mems:
                if isinstance(m.store, VectorStore):
                    if m.store._vectors:
                        vecs.append(np.concatenate(m.store._vectors, axis=0))
                    metas.extend(m.store._meta)
            if vecs:
                base.store = VectorStore(base.store.dim)
                base.store.add(np.concatenate(vecs, axis=0), metas)
    except Exception:
        pass
    return base


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
        scheduler: AcceleratorScheduler | None = None,
        micro_batcher: AdaptiveMicroBatcher | None = None,
        hpc_backend: str | None = None,
        enclave: EnclaveConfig | None = None,
        replay_hook: Callable[[], None] | None = None,
        replay_interval: float | None = None,
        consolidation_hook: Callable[[], None] | None = None,
        consolidation_interval: float | None = None,
        async_mode: bool = False,
        async_workers: int = 2,
        sync_steps: int = 1,

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
        self.replay_hook = replay_hook
        self.replay_interval = replay_interval
        self._last_replay = time.time()
        self.consolidation_hook = consolidation_hook
        self.consolidation_interval = consolidation_interval
        self._last_consolidation = time.time()
        self.async_mode = async_mode
        self.async_workers = max(1, int(async_workers))
        self.sync_steps = max(1, int(sync_steps))

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
                if self.scheduler is not None:
                    if hasattr(self.scheduler, "submit_job"):
                        self.scheduler.submit_job(cmd, backend=self.hpc_backend)
                    elif hasattr(self.scheduler, "submit_best"):
                        self.scheduler.submit_best(cmd)
                    else:
                        submit_job(cmd, backend=self.hpc_backend)
                else:
                    submit_job(cmd, backend=self.hpc_backend)
                self.step += 1
                continue

            if self.async_mode:
                start = self.step
                procs = []
                for wid in range(self.async_workers):
                    p = mp.Process(
                        target=_worker_process,
                        args=(
                            self.train_fn,
                            self.mem_cfg,
                            str(self.checkpoint_dir),
                            start,
                            self.grad_compression,
                            self.enclave_cfg,
                            wid,
                            self.sync_steps,
                        ),
                    )
                    p.start()
                    procs.append(p)
                for p in procs:
                    p.join()
                if any(p.exitcode != 0 for p in procs):
                    restarts += 1
                    if restarts > self.max_restarts:
                        raise RuntimeError("Maximum restarts exceeded")
                    continue
                paths = [
                    self.checkpoint_dir
                    / f"step{start + self.sync_steps}_w{wid}"
                    / "memory"
                    for wid in range(self.async_workers)
                ]
                mems = [DistributedMemory.load(p) for p in paths]
                merged = _average_memories(mems)
                out = self.checkpoint_dir / f"step{start + self.sync_steps}"
                merged.save(out / "memory")
                restarts = 0
                self.step += self.sync_steps
            else:
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
                out = self.checkpoint_dir / f"step{self.step + 1}"
                # memory already saved by worker

            if not self.async_mode:
                self.step += 1
            if self.telemetry:
                stats = self.telemetry.get_stats()
                stats["step"] = self.step
                print("telemetry", stats)
            if self.micro_batcher:
                self.micro_batcher.tick()
            if self.replay_hook is not None and self.replay_interval is not None:
                if time.time() - self._last_replay >= self.replay_interval:
                    self.replay_hook()
                    self._last_replay = time.time()
            if (
                self.consolidation_hook is not None
                and self.consolidation_interval is not None
            ):
                if time.time() - self._last_consolidation >= self.consolidation_interval:
                    self.consolidation_hook()
                    self._last_consolidation = time.time()
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
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--comp-cfg", default=None)
    parser.add_argument("--enclave-cfg", default=None)
    parser.add_argument("--worker-id", type=int, default=0)
    args = parser.parse_args()

    mod_name, fn_name = args.train_fn.rsplit(":", 1)
    module = __import__(mod_name, fromlist=[fn_name])
    fn = getattr(module, fn_name)
    mem_cfg = json.loads(args.mem_cfg)
    comp = json.loads(args.comp_cfg) if args.comp_cfg else None
    encl = json.loads(args.enclave_cfg) if args.enclave_cfg else None
    _worker_process(
        fn,
        mem_cfg,
        args.ckpt_dir,
        args.step,
        comp,
        encl,
        args.worker_id,
        args.steps,
    )

__all__ = [
    "DistributedTrainer",
    "MemoryConfig",
    "GradientCompressionConfig",
    "EnclaveConfig",
]
