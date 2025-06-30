"""gRPC backend to share HierarchicalMemory across nodes."""

from concurrent import futures
from typing import Iterable, Any, Tuple

import grpc
import torch

from .hierarchical_memory import HierarchicalMemory
from . import distributed_memory_pb2 as pb2
from . import distributed_memory_pb2_grpc as pb2_grpc


class _MemoryService(pb2_grpc.MemoryServiceServicer):
    """gRPC servicer wrapping ``HierarchicalMemory``."""

    def __init__(self, memory: HierarchicalMemory) -> None:
        self.memory = memory

    def Push(self, request: pb2.PushRequest, context: grpc.ServicerContext) -> pb2.PushReply:
        vec = torch.tensor(request.vector, dtype=torch.float32)
        meta: Iterable[Any] | None = None
        if request.meta:
            meta = [request.meta]
        self.memory.add(vec, metadata=meta)
        return pb2.PushReply(ok=True)

    def Query(self, request: pb2.QueryRequest, context: grpc.ServicerContext) -> pb2.QueryReply:
        vec = torch.tensor(request.vector, dtype=torch.float32)
        k = request.k if request.k > 0 else 5
        out, meta = self.memory.search(vec, k=k)
        return pb2.QueryReply(
            vectors=out.detach().cpu().reshape(-1).tolist(),
            meta=[str(m) if m is not None else "" for m in meta],
            dim=out.size(-1),
        )


class DistributedMemoryServer:
    """Wrap a ``HierarchicalMemory`` inside a gRPC server."""

    def __init__(self, memory: HierarchicalMemory, address: str) -> None:
        self.memory = memory
        self.address = address
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb2_grpc.add_MemoryServiceServicer_to_server(_MemoryService(memory), self._server)
        self._server.add_insecure_port(address)

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.stop(0)

    def wait(self) -> None:
        self._server.wait_for_termination()


def push_remote(address: str, vector: torch.Tensor, meta: Any | None = None) -> bool:
    """Send ``vector`` to the remote memory server at ``address``."""
    with grpc.insecure_channel(address) as channel:
        stub = pb2_grpc.MemoryServiceStub(channel)
        req = pb2.PushRequest(vector=vector.detach().cpu().tolist(), meta=str(meta) if meta is not None else "")
        resp = stub.Push(req)
    return resp.ok


def query_remote(address: str, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, list[Any]]:
    """Query ``k`` vectors from the remote memory server at ``address``."""
    with grpc.insecure_channel(address) as channel:
        stub = pb2_grpc.MemoryServiceStub(channel)
        req = pb2.QueryRequest(vector=query.detach().cpu().tolist(), k=k)
        resp = stub.Query(req)
    vecs = torch.tensor(resp.vectors, dtype=torch.float32)
    if resp.dim > 0:
        vecs = vecs.view(-1, resp.dim)
    meta = [m for m in resp.meta]
    return vecs, meta

