from __future__ import annotations

from typing import Iterable, Any, Dict, Tuple

import torch

from .hierarchical_memory import (
    HierarchicalMemory,
    MemoryServer,
    query_remote,
)

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional
    _HAS_GRPC = False


if _HAS_GRPC:

    import time
    import uuid

    class FederatedMemoryServer(MemoryServer):
        """Memory server that replicates updates across peers using CRDTs."""

        def __init__(
            self,
            memory: HierarchicalMemory,
            address: str = "localhost:50051",
            peers: Iterable[str] | None = None,
            max_workers: int = 4,
        ) -> None:
            super().__init__(memory, address=address, max_workers=max_workers)
            self.peers = list(peers or [])
            self.state: Dict[str, Tuple[int, torch.Tensor]] = {}

        def add_peer(self, address: str) -> None:
            """Register a new peer."""
            if address not in self.peers:
                self.peers.append(address)

        def remove_peer(self, address: str) -> None:
            """Remove a peer."""
            if address in self.peers:
                self.peers.remove(address)

        # --------------------------------------------------------------
        def _apply_update(self, key: str, vec: torch.Tensor, ts: int) -> None:
            cur = self.state.get(key)
            if cur is not None and cur[0] >= ts:
                return
            if cur is not None:
                self.memory.delete(tag=key)
            self.memory.add(vec.unsqueeze(0), metadata=[key])
            self.state[key] = (ts, vec.detach().cpu())

        def _replicate_entries(self, entries: list[memory_pb2.VectorEntry]) -> None:
            md = (("x-replicated", "1"),)
            req = memory_pb2.SyncRequest(items=entries)
            for addr in self.peers:
                with grpc.insecure_channel(addr) as channel:
                    stub = memory_pb2_grpc.MemoryServiceStub(channel)
                    stub.Sync(req, metadata=md)

        # --------------------------------------------------------------
        def _replicate(self, key: str) -> None:
            ts, vec = self.state[key]
            entry = memory_pb2.VectorEntry(
                id=key,
                vector=vec.view(-1).tolist(),
                metadata=key,
                timestamp=ts,
            )
            self._replicate_entries([entry])

        def _replicate_batch(self, keys: Iterable[str]) -> None:
            entries = []
            for k in keys:
                ts, vec = self.state[k]
                entries.append(
                    memory_pb2.VectorEntry(
                        id=k,
                        vector=vec.view(-1).tolist(),
                        metadata=k,
                        timestamp=ts,
                    )
                )
            if entries:
                self._replicate_entries(entries)

        # gRPC handlers -------------------------------------------------
        def Push(self, request: memory_pb2.PushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            vec = torch.tensor(request.vector).reshape(1, -1)
            meta = request.metadata if request.metadata else None
            key = str(meta) if meta is not None else uuid.uuid4().hex
            ts = int(time.time() * 1000)
            self._apply_update(key, vec[0], ts)
            if not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                self._replicate(key)
            return memory_pb2.PushReply(ok=True)

        def Query(self, request: memory_pb2.QueryRequest, context) -> memory_pb2.QueryReply:  # noqa: N802
            q = torch.tensor(request.vector).reshape(1, -1)
            out, meta = self.memory.search(q, k=int(request.k))
            for addr in self.peers:
                r_vec, r_meta = query_remote(addr, q[0], k=int(request.k))
                if r_vec.numel() > 0:
                    out = torch.cat([out, r_vec.to(q.device)], dim=0)
                    meta.extend(r_meta)
            if out.numel() == 0:
                return memory_pb2.QueryReply(vectors=[], metadata=[])
            scores = out @ q.view(-1, 1)
            idx = torch.argsort(scores.ravel(), descending=True)[: int(request.k)]
            flat = out[idx].detach().cpu().view(-1).tolist()
            meta_out = [str(meta[i]) for i in idx]
            return memory_pb2.QueryReply(vectors=flat, metadata=meta_out)

        def PushBatch(self, request: memory_pb2.PushBatchRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            keys = []
            for item in request.items:
                vec = torch.tensor(item.vector)
                meta = item.metadata if item.metadata else None
                key = str(meta) if meta is not None else uuid.uuid4().hex
                ts = int(time.time() * 1000)
                self._apply_update(key, vec, ts)
                keys.append(key)
            if keys and not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                self._replicate_batch(keys)
            return memory_pb2.PushReply(ok=True)

        def Sync(self, request: memory_pb2.SyncRequest, context) -> memory_pb2.SyncReply:  # noqa: N802
            keys = []
            for item in request.items:
                vec = torch.tensor(item.vector)
                key = item.id or item.metadata
                if not key:
                    continue
                ts = int(item.timestamp)
                self._apply_update(key, vec, ts)
                keys.append(key)
            if keys and not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                self._replicate_batch(keys)
            return memory_pb2.SyncReply(ok=True)

        def start(self) -> None:  # type: ignore[override]
            super().start()

        def stop(self, grace: float = 0) -> None:  # type: ignore[override]
            super().stop(grace)


__all__ = ["FederatedMemoryServer"]

