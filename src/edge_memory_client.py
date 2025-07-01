import torch
from typing import Iterable, Any, Tuple, List

from .remote_memory import RemoteMemory

try:
    import grpc  # type: ignore
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


class EdgeMemoryClient:
    """Buffering client that streams vectors to :class:`RemoteMemory`."""

    def __init__(self, address: str, buffer_size: int = 32) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for EdgeMemoryClient")
        self.remote = RemoteMemory(address)
        self.buffer_size = buffer_size
        self._vec_buf: List[torch.Tensor] = []
        self._meta_buf: List[Any] = []

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
        self.remote.add(batch, self._meta_buf)
        self._vec_buf.clear()
        self._meta_buf.clear()

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[str]]:
        """Flush pending vectors and query the remote store."""
        self.flush()
        return self.remote.search(query, k)

    def close(self) -> None:
        """Flush remaining vectors."""
        self.flush()

    def __enter__(self) -> "EdgeMemoryClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["EdgeMemoryClient"]
