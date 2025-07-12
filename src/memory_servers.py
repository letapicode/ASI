from __future__ import annotations

"""Unified implementations of various gRPC memory servers."""

from concurrent import futures
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Dict
import sys
import time
import uuid
import numpy as np

try:  # optional grpc dependency
    import grpc  # type: ignore
    from . import memory_pb2, memory_pb2_grpc
    from . import fhe_memory_pb2, fhe_memory_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - allow running without grpc
    _HAS_GRPC = False

try:  # optional tenseal dependency
    import tenseal as ts  # type: ignore
    _HAS_TENSEAL = True
except Exception:  # pragma: no cover - optional
    ts = None
    _HAS_TENSEAL = False

if _HAS_GRPC:
    import torch
    from .telemetry import TelemetryLogger
    from .vector_stores import VectorStore
    from .blockchain_provenance_ledger import BlockchainProvenanceLedger
    from .hierarchical_memory import HierarchicalMemory
    from .memory_clients import query_remote
    from .proofs import RetrievalProof, ZKRetrievalProof


    class BaseMemoryServer(memory_pb2_grpc.MemoryServiceServicer):
        """Common gRPC server implementation for memory backends."""

        def __init__(
            self,
            backend: Any,
            address: str = "localhost:50051",
            max_workers: int = 4,
            telemetry: "TelemetryLogger | None" = None,
            *,
            service_adder: "Callable[[Any, grpc.Server], None] | None" = None,
        ) -> None:
            self.backend = backend
            self.address = address
            self.telemetry = telemetry
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
            if service_adder is None:
                service_adder = memory_pb2_grpc.add_MemoryServiceServicer_to_server
            service_adder(self, self.server)
            self.server.add_insecure_port(address)

        # --------------------------------------------------------------
        def Push(self, request: memory_pb2.PushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            vec = torch.tensor(request.vector).reshape(1, -1)
            meta = request.metadata if request.metadata else None
            self.backend.add(vec, metadata=[meta])
            if self.telemetry:
                stats = self.telemetry.get_stats()
                stats["push"] = stats.get("push", 0) + 1
            return memory_pb2.PushReply(ok=True)

        def Query(self, request: memory_pb2.QueryRequest, context) -> memory_pb2.QueryReply:  # noqa: N802
            q = torch.tensor(request.vector).reshape(1, -1)
            out, meta = self.backend.search(q, k=int(request.k))
            flat = out.detach().cpu().view(-1).tolist()
            meta = [str(m) for m in meta]
            if self.telemetry:
                stats = self.telemetry.get_stats()
                stats["query"] = stats.get("query", 0) + 1
            return memory_pb2.QueryReply(vectors=flat, metadata=meta)

        def PushBatch(self, request: memory_pb2.PushBatchRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            for item in request.items:
                vec = torch.tensor(item.vector).reshape(1, -1)
                meta = item.metadata if item.metadata else None
                self.backend.add(vec, metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def QueryBatch(self, request: memory_pb2.QueryBatchRequest, context) -> memory_pb2.QueryBatchReply:  # noqa: N802
            replies = []
            for qreq in request.items:
                qt = torch.tensor(qreq.vector).reshape(1, -1)
                out, meta = self.backend.search(qt, k=int(qreq.k))
                flat = out.detach().cpu().view(-1).tolist()
                meta = [str(m) for m in meta]
                replies.append(memory_pb2.QueryReply(vectors=flat, metadata=meta))
            return memory_pb2.QueryBatchReply(items=replies)

        # --------------------------------------------------------------
        def start(self) -> None:
            if self.telemetry:
                self.telemetry.start()
            self.server.start()

        def stop(self, grace: float = 0) -> None:
            self.server.stop(grace)
            if self.telemetry:
                self.telemetry.stop()


    class MemoryServer(BaseMemoryServer):
        """gRPC server exposing a ``HierarchicalMemory`` backend."""

        def __init__(
            self,
            memory: HierarchicalMemory,
            address: str = "localhost:50051",
            max_workers: int = 4,
            telemetry: "TelemetryLogger | None" = None,
        ) -> None:
            self.memory = memory
            super().__init__(memory, address=address, max_workers=max_workers, telemetry=telemetry)


    class QuantizedMemoryServer(BaseMemoryServer):
        """MemoryServer variant for quantized search."""

        def __init__(
            self,
            memory: HierarchicalMemory,
            address: str = "localhost:50120",
            max_workers: int = 4,
            telemetry: "TelemetryLogger | None" = None,
        ) -> None:
            self.memory = memory
            super().__init__(memory, address=address, max_workers=max_workers, telemetry=telemetry)


    class QuantumMemoryServer(BaseMemoryServer):
        """Expose a :class:`VectorStore` with quantum search."""

        def __init__(self, store: VectorStore, address: str = "localhost:50520", max_workers: int = 4) -> None:
            self.store = store
            super().__init__(store, address=address, max_workers=max_workers)

        # --------------------------------------------------------------
        def Push(self, request: memory_pb2.PushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            vec = np.asarray(request.vector, dtype=np.float32)
            meta = request.metadata if request.metadata else None
            self.store.add(vec.reshape(1, -1), metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def Query(self, request: memory_pb2.QueryRequest, context) -> memory_pb2.QueryReply:  # noqa: N802
            q = np.asarray(request.vector, dtype=np.float32)
            out, meta = self.store.search(q, k=int(request.k), quantum=True)
            return memory_pb2.QueryReply(
                vectors=out.reshape(-1).tolist(),
                metadata=["" if m is None else str(m) for m in meta],
            )

        def PushBatch(self, request: memory_pb2.PushBatchRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            vecs: list[np.ndarray] = []
            metas: list[Any] = []
            for item in request.items:
                vecs.append(np.asarray(item.vector, dtype=np.float32))
                metas.append(item.metadata if item.metadata else None)
            if vecs:
                arr = np.stack(vecs, axis=0)
                self.store.add(arr, metadata=metas)
            return memory_pb2.PushReply(ok=True)

        def QueryBatch(self, request: memory_pb2.QueryBatchRequest, context) -> memory_pb2.QueryBatchReply:  # noqa: N802
            replies = []
            for q in request.items:
                vec = np.asarray(q.vector, dtype=np.float32)
                out, meta = self.store.search(vec, k=int(q.k), quantum=True)
                replies.append(
                    memory_pb2.QueryReply(
                        vectors=out.reshape(-1).tolist(),
                        metadata=["" if m is None else str(m) for m in meta],
                    )
                )
            return memory_pb2.QueryBatchReply(items=replies)


    class ZeroTrustMemoryServer(BaseMemoryServer):
        """Memory server that verifies signed access tokens."""

        def __init__(
            self,
            memory: Any,
            ledger: BlockchainProvenanceLedger,
            address: str = "localhost:50051",
            max_workers: int = 4,
            telemetry: "TelemetryLogger | None" = None,
        ) -> None:
            self.memory = memory
            self.ledger = ledger
            super().__init__(memory, address=address, max_workers=max_workers, telemetry=telemetry)

        # --------------------------------------------------
        def _check_token(self, context: grpc.ServicerContext) -> bool:
            md = dict(context.invocation_metadata())
            token = md.get("authorization")
            if not token:
                return False
            try:
                return self.ledger.verify([token])
            except Exception:
                return False

        # --------------------------------------------------
        def Push(self, request: memory_pb2.PushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            vec = torch.tensor(request.vector).reshape(1, -1)
            meta = request.metadata if request.metadata else None
            self.memory.add(vec, metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def Query(self, request: memory_pb2.QueryRequest, context) -> memory_pb2.QueryReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            q = torch.tensor(request.vector).reshape(1, -1)
            out, meta = self.memory.search(q, k=int(request.k))
            flat = out.detach().cpu().view(-1).tolist()
            meta = [str(m) for m in meta]
            return memory_pb2.QueryReply(vectors=flat, metadata=meta)

        def PushBatch(self, request: memory_pb2.PushBatchRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            for item in request.items:
                vec = torch.tensor(item.vector).reshape(1, -1)
                meta = item.metadata if item.metadata else None
                self.memory.add(vec, metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def QueryBatch(self, request: memory_pb2.QueryBatchRequest, context) -> memory_pb2.QueryBatchReply:  # noqa: N802
            if not self._check_token(context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
            replies = []
            for qreq in request.items:
                qt = torch.tensor(qreq.vector).reshape(1, -1)
                out, meta = self.memory.search(qt, k=int(qreq.k))
                flat = out.detach().cpu().view(-1).tolist()
                meta = [str(m) for m in meta]
                replies.append(memory_pb2.QueryReply(vectors=flat, metadata=meta))
            return memory_pb2.QueryBatchReply(items=replies)


    class FHEMemoryServer(BaseMemoryServer, fhe_memory_pb2_grpc.FHEMemoryServiceServicer):
        """gRPC server operating on encrypted vectors via TenSEAL."""

        def __init__(self, store: VectorStore, ctx: "ts.Context", address: str = "localhost:50900", max_workers: int = 4) -> None:
            if not _HAS_TENSEAL:
                raise ImportError("tenseal is required for FHEMemoryServer")
            self.store = store
            self.ctx = ctx
            super().__init__(
                store,
                address=address,
                max_workers=max_workers,
                service_adder=fhe_memory_pb2_grpc.add_FHEMemoryServiceServicer_to_server,
            )

        # --------------------------------------------------
        def Push(self, request: fhe_memory_pb2.FHEPushRequest, context) -> memory_pb2.PushReply:  # noqa: N802
            enc = ts.CKKSVector.load(self.ctx, request.vector)
            vec = np.array(enc.decrypt(), dtype=np.float32)
            meta = request.metadata if request.metadata else None
            self.store.add(vec.reshape(1, -1), metadata=[meta])
            return memory_pb2.PushReply(ok=True)

        def Query(self, request: fhe_memory_pb2.FHEQueryRequest, context) -> fhe_memory_pb2.FHEQueryReply:  # noqa: N802
            enc_q = ts.CKKSVector.load(self.ctx, request.vector)
            q = np.array(enc_q.decrypt(), dtype=np.float32)
            out, meta = self.store.search(q, k=int(request.k))
            enc_out = ts.ckks_vector(self.ctx, out.reshape(-1).tolist())
            proof = ZKRetrievalProof.generate(out, ["" if m is None else str(m) for m in meta])
            return fhe_memory_pb2.FHEQueryReply(
                vectors=enc_out.serialize(),
                metadata=["" if m is None else str(m) for m in meta],
                proof=proof.digest,
            )

        # start/stop inherited from ``BaseMemoryServer``


    class FHEMemoryClient:
        """Thin client for :class:`FHEMemoryServer`."""

        def __init__(self, address: str, ctx: "ts.Context") -> None:
            if not _HAS_TENSEAL:
                raise ImportError("tenseal is required for FHEMemoryClient")
            self.ctx = ctx
            self.channel = grpc.insecure_channel(address)
            self.stub = fhe_memory_pb2_grpc.FHEMemoryServiceStub(self.channel)

        def add(self, vector: np.ndarray, metadata: Any | None = None) -> None:
            enc = ts.ckks_vector(self.ctx, np.asarray(vector, dtype=np.float32).ravel().tolist())
            req = fhe_memory_pb2.FHEPushRequest(
                vector=enc.serialize(),
                metadata="" if metadata is None else str(metadata),
            )
            self.stub.Push(req)

        def search(self, vector: np.ndarray, k: int = 5, verify: bool = True):
            enc_q = ts.ckks_vector(self.ctx, np.asarray(vector, dtype=np.float32).ravel().tolist())
            req = fhe_memory_pb2.FHEQueryRequest(vector=enc_q.serialize(), k=k)
            reply = self.stub.Query(req)
            enc_out = ts.CKKSVector.load(self.ctx, reply.vectors)
            dim = vector.size
            out = np.array(enc_out.decrypt(), dtype=np.float32).reshape(-1, dim)
            proof = ZKRetrievalProof(reply.proof)
            if verify and not proof.verify(out, reply.metadata):
                raise ValueError("invalid retrieval proof")
            return out, list(reply.metadata)

        def close(self) -> None:
            self.channel.close()


    class FederatedMemoryServer(BaseMemoryServer):
        """Memory server that replicates updates across peers using CRDTs."""

        def __init__(
            self,
            memory: HierarchicalMemory,
            address: str = "localhost:50051",
            peers: Iterable[str] | None = None,
            max_workers: int = 4,
            *,
            require_proof: bool = False,
        ) -> None:
            self.memory = memory
            super().__init__(memory, address=address, max_workers=max_workers)
            self.peers = list(peers or [])
            self.state: Dict[str, _VectorState] = {}
            self.require_proof = require_proof

        @dataclass
        class _VectorState:
            ts: int
            vec: torch.Tensor
            digest: str

        def add_peer(self, address: str) -> None:
            if address not in self.peers:
                self.peers.append(address)

        def remove_peer(self, address: str) -> None:
            if address in self.peers:
                self.peers.remove(address)

        # --------------------------------------------------------------
        def _apply_update(self, key: str, vec: torch.Tensor, ts: int) -> None:
            cur = self.state.get(key)
            if cur is not None and cur.ts >= ts:
                return
            if cur is not None:
                self.memory.delete(tag=key)
            self.memory.add(vec.unsqueeze(0), metadata=[key])
            digest = RetrievalProof.generate(vec).digest
            self.state[key] = FederatedMemoryServer._VectorState(ts, vec.detach().cpu(), digest)

        def _replicate_entries(self, entries: list[memory_pb2.VectorEntry]) -> None:
            md = (("x-replicated", "1"),)
            req = memory_pb2.SyncRequest(items=entries)

            def _send(addr: str) -> None:
                with grpc.insecure_channel(addr) as channel:
                    stub = memory_pb2_grpc.MemoryServiceStub(channel)
                    stub.Sync(req, metadata=md)

            with futures.ThreadPoolExecutor(max_workers=len(self.peers)) as exe:
                futs = [exe.submit(_send, addr) for addr in self.peers]
                for f in futs:
                    f.result()

        # --------------------------------------------------------------
        def _replicate(self, key: str) -> None:
            state = self.state[key]
            entry = memory_pb2.VectorEntry(
                id=key,
                vector=state.vec.view(-1).tolist(),
                metadata=key,
                timestamp=state.ts,
                proof=state.digest,
            )
            self._replicate_entries([entry])

        def _replicate_batch(self, keys: Iterable[str]) -> None:
            entries = []
            for k in keys:
                state = self.state[k]
                entries.append(
                    memory_pb2.VectorEntry(
                        id=k,
                        vector=state.vec.view(-1).tolist(),
                        metadata=k,
                        timestamp=state.ts,
                        proof=state.digest,
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
            if keys and not any(
                m.key == "x-replicated" for m in context.invocation_metadata()
            ):
                self._replicate_batch(keys)
            return memory_pb2.PushReply(ok=True)

        def Sync(self, request: memory_pb2.SyncRequest, context) -> memory_pb2.SyncReply:  # noqa: N802
            keys = []
            for item in request.items:
                vec = torch.tensor(item.vector)
                key = item.id or item.metadata
                if not key:
                    continue
                if self.require_proof:
                    if not item.proof or not RetrievalProof(item.proof).verify(vec):
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("invalid retrieval proof")
                        return memory_pb2.SyncReply(ok=False)
                ts = int(item.timestamp)
                self._apply_update(key, vec, ts)
                keys.append(key)
            if keys and not any(
                m.key == "x-replicated" for m in context.invocation_metadata()
            ):
                self._replicate_batch(keys)
            return memory_pb2.SyncReply(ok=True)

    __all__ = [
        "BaseMemoryServer",
        "MemoryServer",
        "QuantizedMemoryServer",
        "QuantumMemoryServer",
        "ZeroTrustMemoryServer",
        "FHEMemoryServer",
        "FHEMemoryClient",
        "FederatedMemoryServer",
    ]
else:  # fallback definitions when grpc is unavailable
    BaseMemoryServer = None  # type: ignore
    MemoryServer = None  # type: ignore
    QuantizedMemoryServer = None  # type: ignore
    QuantumMemoryServer = None  # type: ignore
    ZeroTrustMemoryServer = None  # type: ignore
    FHEMemoryServer = None  # type: ignore
    FHEMemoryClient = None  # type: ignore
    FederatedMemoryServer = None  # type: ignore
    __all__ = [
        "BaseMemoryServer",
        "MemoryServer",
        "QuantizedMemoryServer",
        "QuantumMemoryServer",
        "ZeroTrustMemoryServer",
        "FHEMemoryServer",
        "FHEMemoryClient",
        "FederatedMemoryServer",
    ]
