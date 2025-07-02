from __future__ import annotations

from typing import Iterable, Any

import torch

from .hierarchical_memory import (
    HierarchicalMemory,
    MemoryServer,
    push_remote,
    push_batch_remote,
    query_remote,
)

try:
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional
    _HAS_GRPC = False


if _HAS_GRPC:

    class FederatedMemoryServer(MemoryServer):
        """Memory server that replicates updates across peers."""

        def __init__(
            self,
            memory: HierarchicalMemory,
            address: str = "localhost:50051",
            peers: Iterable[str] | None = None,
            max_workers: int = 4,
        ) -> None:
            super().__init__(memory, address=address, max_workers=max_workers)
            self.peers = list(peers or [])

        def add_peer(self, address: str) -> None:
            """Register a new peer."""
            if address not in self.peers:
                self.peers.append(address)

        def remove_peer(self, address: str) -> None:
            """Remove a peer."""
            if address in self.peers:
                self.peers.remove(address)

        # --------------------------------------------------------------
        def _replicate(self, vector: torch.Tensor, meta: Any | None) -> None:
            for addr in self.peers:
                push_remote(addr, vector, meta)

        def _replicate_batch(
            self, vectors: torch.Tensor, metas: Iterable[Any] | None
        ) -> None:
            for addr in self.peers:
                push_batch_remote(addr, vectors, metas)

        # gRPC handlers -------------------------------------------------
        def Push(self, request: memory_pb2.PushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            vec = torch.tensor(request.vector).reshape(1, -1)
            meta = request.metadata if request.metadata else None
            self.memory.add(vec, metadata=[meta])
            if not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                self._replicate(vec[0], meta)
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
            vectors = []
            metas = []
            for item in request.items:
                vectors.append(torch.tensor(item.vector))
                metas.append(item.metadata if item.metadata else None)
            if vectors:
                vec_batch = torch.stack(vectors)
                self.memory.add(vec_batch, metadata=metas)
                if not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                    self._replicate_batch(vec_batch, metas)
            return memory_pb2.PushReply(ok=True)

        def start(self) -> None:  # type: ignore[override]
            super().start()

        def stop(self, grace: float = 0) -> None:  # type: ignore[override]
            super().stop(grace)


__all__ = ["FederatedMemoryServer"]
