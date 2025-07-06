import torch
from typing import Iterable, Any, Tuple, List, Deque
from collections import deque
import threading
import time

from .remote_memory import RemoteMemory

try:
    import grpc  # type: ignore
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


class EdgeMemoryClient:
    """Buffering client that streams vectors to :class:`RemoteMemory`."""

    def __init__(self, address: str, buffer_size: int = 32, sync_interval: float = 2.0) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for EdgeMemoryClient")
        self.remote = RemoteMemory(address)
        self.buffer_size = buffer_size
        self.sync_interval = sync_interval
        self._vec_buf: List[torch.Tensor] = []
        self._meta_buf: List[Any] = []
        self._queue: Deque[tuple[str, torch.Tensor | None, Any | None]] = deque()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()

    # --------------------------------------------------------------
    def _sync_loop(self) -> None:
        while not self._stop.is_set():
            if self._queue:
                try:
                    self._flush_queue()
                except Exception:  # pragma: no cover - network failure
                    pass
            time.sleep(self.sync_interval)

    def _flush_queue(self) -> None:
        ops = list(self._queue)
        self._queue.clear()
        for op, vec, meta in ops:
            if op == "add" and vec is not None:
                self.remote.add(vec.unsqueeze(0), [meta])
            elif op == "delete" and hasattr(self.remote, "delete"):
                try:
                    self.remote.delete(tag=meta)
                except Exception:
                    self._queue.append((op, vec, meta))

    def add(self, x: torch.Tensor, metadata: Iterable[Any] | None = None) -> None:
        """Add vectors to the send buffer and flush when full."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        metas = list(metadata) if metadata is not None else [None] * len(x)
        for vec, meta in zip(x, metas):
            self._vec_buf.append(vec)
            self._meta_buf.append(meta)
            if len(self._vec_buf) >= self.buffer_size:
                self.flush()

    def flush(self) -> None:
        """Send buffered vectors to the remote store."""
        if not self._vec_buf:
            return
        batch = torch.stack(self._vec_buf)
        try:
            self.remote.add(batch, self._meta_buf)
        except Exception:  # pragma: no cover - network failure
            for vec, meta in zip(self._vec_buf, self._meta_buf):
                self._queue.append(("add", vec, meta))
        else:
            self._vec_buf.clear()
            self._meta_buf.clear()

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[str]]:
        """Flush pending vectors and query the remote store."""
        self.flush()
        return self.remote.search(query, k)

    def delete(self, *, tag: Any) -> None:
        """Delete vectors by tag, queueing on failure."""
        try:
            if hasattr(self.remote, "delete"):
                self.remote.delete(tag=tag)
        except Exception:  # pragma: no cover - network failure
            self._queue.append(("delete", None, tag))

    def close(self) -> None:
        """Flush remaining vectors."""
        self.flush()
        self._stop.set()
        self._thread.join(timeout=0.1)
        if self._queue:
            try:
                self._flush_queue()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> "EdgeMemoryClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["EdgeMemoryClient"]
