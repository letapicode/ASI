"""Deprecated: use :mod:`asi.memory_clients` instead."""

from .memory_clients import (
    RemoteMemoryClient as RemoteMemory,
    push_remote,
    query_remote,
    push_remote_async,
    query_remote_async,
    push_batch_remote,
    query_batch_remote,
    push_batch_remote_async,
    query_batch_remote_async,
)

__all__ = [
    "RemoteMemory",
    "push_remote",
    "query_remote",
    "push_remote_async",
    "query_remote_async",
    "push_batch_remote",
    "query_batch_remote",
    "push_batch_remote_async",
    "query_batch_remote_async",
]
