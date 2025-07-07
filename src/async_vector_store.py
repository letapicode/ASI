import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Iterable, Any, Tuple

import numpy as np

from .vector_store import FaissVectorStore


class AsyncFaissVectorStore(FaissVectorStore):
    """FAISS vector store with async add/search using threads."""

    def __init__(self, dim: int, path: str | Path | None = None, workers: int = 2) -> None:
        super().__init__(dim=dim, path=path)
        self._executor = ThreadPoolExecutor(max_workers=workers)

    def add_async(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> Future:
        """Schedule ``add`` on a background thread."""
        return self._executor.submit(super().add, vectors, metadata)

    def search_async(self, query: np.ndarray, k: int = 5) -> Future:
        """Schedule ``search`` on a background thread."""
        return self._executor.submit(super().search, query, k)

    # ------------------------------------------------------------------
    # HyDE search helpers

    def hyde_search_async(self, query: np.ndarray, k: int = 5) -> Future:
        """Schedule ``hyde_search`` on a background thread."""
        return self._executor.submit(super().hyde_search, query, k)

    async def ahyde_search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, list[Any]]:
        """Awaitable ``hyde_search`` wrapper using ``asyncio``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, super().hyde_search, query, k)

    async def save_async(self, path: str | Path) -> None:
        """Awaitable ``save`` wrapper using ``asyncio``."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, super().save, path)

    @classmethod
    async def load_async(cls, path: str | Path) -> "AsyncFaissVectorStore":
        """Awaitable ``load`` wrapper using ``asyncio``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, cls.load, path)

    async def aadd(self, vectors: np.ndarray, metadata: Iterable[Any] | None = None) -> None:
        """Awaitable ``add`` wrapper using ``asyncio``."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, super().add, vectors, metadata)

    async def asearch(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, list[Any]]:
        """Awaitable ``search`` wrapper using ``asyncio``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, super().search, query, k)

    def close(self) -> None:
        """Shut down the thread pool and persist to disk if needed."""
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "AsyncFaissVectorStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aenter__(self) -> "AsyncFaissVectorStore":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit the async context manager and close resources."""
        self.close()
