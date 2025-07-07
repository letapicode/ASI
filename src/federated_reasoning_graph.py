from __future__ import annotations

import ast
import time
import uuid
from typing import Iterable, Dict, Tuple, List

from .graph_of_thought import GraphOfThought

try:
    import grpc  # type: ignore
    from concurrent import futures
    from . import reasoning_graph_pb2, reasoning_graph_pb2_grpc

    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional
    _HAS_GRPC = False


if _HAS_GRPC:

    class FederatedReasoningGraph(reasoning_graph_pb2_grpc.ReasoningGraphServiceServicer):
        """Replicate :class:`GraphOfThought` nodes across peers using CRDT merges."""

        def __init__(
            self,
            graph: GraphOfThought,
            address: str = "localhost:53051",
            peers: Iterable[str] | None = None,
            max_workers: int = 4,
        ) -> None:
            self.graph = graph
            self.address = address
            self.peers = list(peers or [])
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
            reasoning_graph_pb2_grpc.add_ReasoningGraphServiceServicer_to_server(self, self.server)
            self.server.add_insecure_port(address)
            self.state: Dict[str, Tuple[int, str, List[str], dict]] = {}
            self.id_map: Dict[str, int] = {}

        # ------------------------------------------------------------------
        def add_peer(self, address: str) -> None:
            if address not in self.peers:
                self.peers.append(address)

        def remove_peer(self, address: str) -> None:
            if address in self.peers:
                self.peers.remove(address)

        # ------------------------------------------------------------------
        def _apply_update(self, key: str, text: str, edges: list[str], meta: dict, ts: int) -> None:
            cur = self.state.get(key)
            if cur is not None and cur[0] >= ts:
                return
            if key not in self.id_map:
                nid = self.graph.add_step(text, metadata=meta)
                self.id_map[key] = nid
            else:
                nid = self.id_map[key]
                self.graph.nodes[nid].text = text
                self.graph.nodes[nid].metadata = meta
            self.graph.edges[nid] = [self.id_map[e] for e in edges if e in self.id_map]
            self.state[key] = (ts, text, edges, meta)

        def _replicate_entries(self, entries: list[reasoning_graph_pb2.NodeEntry]) -> None:
            md = (("x-replicated", "1"),)
            req = reasoning_graph_pb2.NodeBatch(items=entries)
            for addr in self.peers:
                with grpc.insecure_channel(addr) as channel:
                    stub = reasoning_graph_pb2_grpc.ReasoningGraphServiceStub(channel)
                    stub.Sync(req, metadata=md)

        def _replicate(self, key: str) -> None:
            ts, text, edges, meta = self.state[key]
            entry = reasoning_graph_pb2.NodeEntry(
                id=key,
                text=text,
                edges=edges,
                metadata=str(meta),
                timestamp=ts,
            )
            self._replicate_entries([entry])

        def _replicate_batch(self, keys: Iterable[str]) -> None:
            entries = []
            for k in keys:
                ts, text, edges, meta = self.state[k]
                entries.append(
                    reasoning_graph_pb2.NodeEntry(
                        id=k,
                        text=text,
                        edges=edges,
                        metadata=str(meta),
                        timestamp=ts,
                    )
                )
            if entries:
                self._replicate_entries(entries)

        # gRPC handlers -------------------------------------------------
        def Push(self, request: reasoning_graph_pb2.NodeEntry, context) -> reasoning_graph_pb2.PushReply:  # noqa: N802
            key = request.id or uuid.uuid4().hex
            meta = ast.literal_eval(request.metadata) if request.metadata else {}
            self._apply_update(key, request.text, list(request.edges), meta, int(time.time() * 1000))
            if not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                self._replicate(key)
            return reasoning_graph_pb2.PushReply(ok=True)

        def PushBatch(self, request: reasoning_graph_pb2.NodeBatch, context) -> reasoning_graph_pb2.PushReply:  # noqa: N802
            keys = []
            for item in request.items:
                key = item.id or uuid.uuid4().hex
                meta = ast.literal_eval(item.metadata) if item.metadata else {}
                self._apply_update(key, item.text, list(item.edges), meta, int(time.time() * 1000))
                keys.append(key)
            if keys and not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                self._replicate_batch(keys)
            return reasoning_graph_pb2.PushReply(ok=True)

        def Sync(self, request: reasoning_graph_pb2.NodeBatch, context) -> reasoning_graph_pb2.PushReply:  # noqa: N802
            keys = []
            for item in request.items:
                key = item.id or uuid.uuid4().hex
                meta = ast.literal_eval(item.metadata) if item.metadata else {}
                self._apply_update(key, item.text, list(item.edges), meta, int(item.timestamp))
                keys.append(key)
            if keys and not any(m.key == "x-replicated" for m in context.invocation_metadata()):
                self._replicate_batch(keys)
            return reasoning_graph_pb2.PushReply(ok=True)

        def start(self) -> None:
            self.server.start()

        def stop(self, grace: float = 0) -> None:
            self.server.stop(grace)


    def push_node_remote(
        address: str,
        text: str,
        edges: Iterable[str] | None = None,
        metadata: dict | None = None,
        key: str | None = None,
        timeout: float = 5.0,
    ) -> bool:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for remote reasoning graph")
        with grpc.insecure_channel(address) as channel:
            stub = reasoning_graph_pb2_grpc.ReasoningGraphServiceStub(channel)
            req = reasoning_graph_pb2.NodeEntry(
                id="" if key is None else key,
                text=text,
                edges=list(edges or []),
                metadata="" if metadata is None else str(metadata),
                timestamp=0,
            )
            reply = stub.Push(req, timeout=timeout)
            return reply.ok

    def sync_nodes_remote(
        address: str,
        entries: Iterable[tuple[str, str, list[str], dict, int]],
        timeout: float = 5.0,
    ) -> bool:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for remote reasoning graph")
        with grpc.insecure_channel(address) as channel:
            stub = reasoning_graph_pb2_grpc.ReasoningGraphServiceStub(channel)
            items = [
                reasoning_graph_pb2.NodeEntry(
                    id=e[0],
                    text=e[1],
                    edges=e[2],
                    metadata=str(e[3]),
                    timestamp=e[4],
                )
                for e in entries
            ]
            req = reasoning_graph_pb2.NodeBatch(items=items)
            reply = stub.Sync(req, timeout=timeout)
            return reply.ok


__all__ = ["FederatedReasoningGraph", "push_node_remote", "sync_nodes_remote"]
