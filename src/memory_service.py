from concurrent import futures
import torch
import grpc

from .hierarchical_memory import HierarchicalMemory
from . import memory_pb2, memory_pb2_grpc


class MemoryService(memory_pb2_grpc.MemoryServiceServicer):
    """gRPC wrapper around ``HierarchicalMemory``."""

    def __init__(self, memory: HierarchicalMemory) -> None:
        self.memory = memory

    def Add(self, request: memory_pb2.AddRequest, context):
        if request.vectors:
            data = torch.tensor([v.values for v in request.vectors], dtype=torch.float32)
            metas = list(request.metadata) if request.metadata else None
            self.memory.add(data, metas)
        return memory_pb2.Empty()

    def Query(self, request: memory_pb2.QueryRequest, context):
        q = torch.tensor(request.query.values, dtype=torch.float32)
        out, meta = self.memory.search(q, k=request.k)
        vectors = [memory_pb2.Vector(values=o.tolist()) for o in out]
        return memory_pb2.QueryResult(vectors=vectors, metadata=[str(m) for m in meta])


def serve(memory: HierarchicalMemory, address: str) -> grpc.Server:
    """Start the gRPC memory server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    memory_pb2_grpc.add_MemoryServiceServicer_to_server(MemoryService(memory), server)
    server.add_insecure_port(address)
    server.start()
    return server
