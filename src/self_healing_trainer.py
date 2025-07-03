"""Simple trainer that restarts failed distributed workers."""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Any
from .training_anomaly_detector import TrainingAnomalyDetector


class SelfHealingTrainer:
    """Run a distributed training function with automatic restarts."""

    def __init__(
        self,
        worker_fn: Callable[[int, int, Any], float | None],
        world_size: int,
        max_restarts: int = 1,
        anomaly_detector: TrainingAnomalyDetector | None = None,
        **worker_kwargs: Any,
    ) -> None:
        self.worker_fn = worker_fn
        self.world_size = world_size
        self.max_restarts = max_restarts
        self.worker_kwargs = worker_kwargs
        self.anomaly_detector = anomaly_detector
        self._restarts = [0] * world_size
        self._procs: list[mp.Process | None] = [None] * world_size

    def _worker_wrapper(self, rank: int, queue: mp.Queue) -> None:
        try:
            loss = self.worker_fn(rank, self.world_size, **self.worker_kwargs)
            queue.put((rank, 0, 0.0 if loss is None else float(loss)))
        except Exception:  # pragma: no cover - runtime failure path
            queue.put((rank, 1, 0.0))

    def _start_worker(self, rank: int, queue: mp.Queue) -> None:
        proc = mp.Process(target=self._worker_wrapper, args=(rank, queue))
        proc.start()
        self._procs[rank] = proc

    def run(self) -> None:
        """Launch workers and restart them if they fail."""
        queue: mp.Queue = mp.Queue()
        for r in range(self.world_size):
            self._start_worker(r, queue)
        finished = 0
        while finished < self.world_size:
            rank, status, loss = queue.get()
            proc = self._procs[rank]
            if proc is not None:
                proc.join()
            if status != 0 and self._restarts[rank] < self.max_restarts:
                self._restarts[rank] += 1
                self._start_worker(rank, queue)
                continue
            if status == 0 and self.anomaly_detector is not None:
                if self.anomaly_detector.record(loss):
                    if self._restarts[rank] < self.max_restarts:
                        self._restarts[rank] += 1
                        self._start_worker(rank, queue)
                        continue
                    else:
                        status = 1
            if status != 0:
                for p in self._procs:
                    if p is not None and p.is_alive():
                        p.terminate()
                raise RuntimeError(f"worker {rank} failed")
            finished += 1
        for p in self._procs:
            if p is not None:
                p.join()


__all__ = ["SelfHealingTrainer"]
