from __future__ import annotations

import torch
from typing import Iterable, Tuple, List, Any

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    from .memory_client_base import MemoryClientBase
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False
    MemoryClientBase = object  # type: ignore


class QuantizedMemoryClient(MemoryClientBase):
    """Thin client for :class:`QuantizedMemoryServer`."""

    def __init__(self, address: str) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for QuantizedMemoryClient")
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)



__all__ = ["QuantizedMemoryClient"]
