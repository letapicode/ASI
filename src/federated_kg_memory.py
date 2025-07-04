from __future__ import annotations

from typing import Iterable, Tuple, Dict, List
import time
import uuid

from .knowledge_graph_memory import KnowledgeGraphMemory

try:
    import grpc  # type: ignore
    from . import kg_memory_pb2, kg_memory_pb2_grpc
    from concurrent import futures

    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


if _HAS_GRPC:

    class FederatedKGMemoryServer(kg_memory_pb2_grpc.KGMemoryServiceServicer):
        """Replicate :class:`KnowledgeGraphMemory` updates across peers using CRDTs."""

        def __init__(
            self,
            memory: KnowledgeGraphMemory,
            address: str = "localhost:52051",
            peers: Iterable[str] | None = None,
            max_workers: int = 4,
        ) -> None:
            self.memory = memory
            self.address = address
            self.peers = list(peers or [])
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
            kg_memory_pb2_grpc.add_KGMemoryServiceServicer_to_server(self, self.server)
            self.server.add_insecure_port(address)
            self.state: Dict[str, Tuple[int, Tuple[str, str, str]]] = {}

        # ------------------------------------------------------------------
        def add_peer(self, address: str) -> None:
            if address not in self.peers:
                self.peers.append(address)

        def remove_peer(self, address: str) -> None:
            if address in self.peers:
                self.peers.remove(address)

        # ------------------------------------------------------------------
        def _apply_update(self, key: str, triple: Tuple[str, str, str], ts: int) -> None:
            cur = self.state.get(key)
            if cur is not None and cur[0] >= ts:
                return
            if cur is None:
                self.memory.add_triples([triple])
            else:
                # overwrite old triple
                self.memory.triples[self.memory.triples.index(cur[1])] = triple
            self.state[key] = (ts, triple)

        def _replicate_entries(self, entries: List[kg_memory_pb2.TripleEntry]) -> None:
            md = (("x-replicated", "1"),)
            req = kg_memory_pb2.TripleBatch(items=entries)
            for addr in self.peers:
                with grpc.insecure_channel(addr) as channel:
                    stub = kg_memory_pb2_grpc.KGMemoryServiceStub(channel)
                    stub.Sync(req, metadata=md)

        def _replicate(self, key: str) -> None:
            ts, triple = self.state[key]
            entry = kg_memory_pb2.TripleEntry(
                id=key,
                subject=triple[0],
                predicate=triple[1],
                object=triple[2],
                timestamp=ts,
            )
            self._replicate_entries([entry])

        def _replicate_batch(self, keys: Iterable[str]) -> None:
            entries: List[kg_memory_pb2.TripleEntry] = []
            for k in keys:
                ts, triple = self.state[k]
                entries.append(
                    kg_memory_pb2.TripleEntry(
                        id=k,
                        subject=triple[0],
                        predicate=triple[1],
                        object=triple[2],
                        timestamp=ts,
                    )
                )
            if entries:
                self._replicate_entries(entries)

        # gRPC handlers -----------------------------------------------------
        def Push(self, request: kg_memory_pb2.TripleEntry, context) -> kg_memory_pb2.PushReply:  # noqa: N802
            triple = (request.subject, request.predicate, request.object)
            key = request.id or uuid.uuid4().hex
            ts = int(time.time() * 1000)
            self._apply_update(key, triple, ts)
            if not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                self._replicate(key)
            return kg_memory_pb2.PushReply(ok=True)

        def PushBatch(self, request: kg_memory_pb2.TripleBatch, context) -> kg_memory_pb2.PushReply:  # noqa: N802
            keys = []
            for item in request.items:
                triple = (item.subject, item.predicate, item.object)
                key = item.id or uuid.uuid4().hex
                ts = int(time.time() * 1000)
                self._apply_update(key, triple, ts)
                keys.append(key)
            if keys and not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                self._replicate_batch(keys)
            return kg_memory_pb2.PushReply(ok=True)

        def Query(self, request: kg_memory_pb2.TripleQuery, context) -> kg_memory_pb2.TripleBatch:  # noqa: N802
            triples = self.memory.query_triples(
                subject=request.subject or None,
                predicate=request.predicate or None,
                object=request.object or None,
            )
            items = []
            for t in triples:
                key = "|".join(t)
                ts = self.state.get(key, (0,))[0]
                items.append(
                    kg_memory_pb2.TripleEntry(
                        id=key,
                        subject=t[0],
                        predicate=t[1],
                        object=t[2],
                        timestamp=ts,
                    )
                )
            return kg_memory_pb2.TripleBatch(items=items)

        def Sync(self, request: kg_memory_pb2.TripleBatch, context) -> kg_memory_pb2.PushReply:  # noqa: N802
            keys = []
            for item in request.items:
                triple = (item.subject, item.predicate, item.object)
                key = item.id or "|".join(triple)
                ts = int(item.timestamp)
                self._apply_update(key, triple, ts)
                keys.append(key)
            if keys and not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                self._replicate_batch(keys)
            return kg_memory_pb2.PushReply(ok=True)

        def start(self) -> None:
            self.server.start()

        def stop(self, grace: float = 0) -> None:
            self.server.stop(grace)


    def push_triple_remote(
        address: str, triple: Tuple[str, str, str], key: str | None = None, timeout: float = 5.0
    ) -> bool:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for remote KG memory")
        with grpc.insecure_channel(address) as channel:
            stub = kg_memory_pb2_grpc.KGMemoryServiceStub(channel)
            req = kg_memory_pb2.TripleEntry(
                id="" if key is None else key,
                subject=triple[0],
                predicate=triple[1],
                object=triple[2],
                timestamp=0,
            )
            reply = stub.Push(req, timeout=timeout)
            return reply.ok

    def query_triples_remote(
        address: str,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        timeout: float = 5.0,
    ) -> List[Tuple[str, str, str]]:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for remote KG memory")
        with grpc.insecure_channel(address) as channel:
            stub = kg_memory_pb2_grpc.KGMemoryServiceStub(channel)
            req = kg_memory_pb2.TripleQuery(
                subject="" if subject is None else subject,
                predicate="" if predicate is None else predicate,
                object="" if object is None else object,
            )
            reply = stub.Query(req, timeout=timeout)
            return [(i.subject, i.predicate, i.object) for i in reply.items]


__all__ = ["FederatedKGMemoryServer", "push_triple_remote", "query_triples_remote"]
