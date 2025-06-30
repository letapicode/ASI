import torch
import grpc
from typing import Iterable, Tuple, List, Any

from . import memory_pb2, memory_pb2_grpc


class RemoteMemory:
    """Thin client for the gRPC memory service."""

    def __init__(self, address: str) -> None:
        self.channel = grpc.insecure_channel(address)
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)

    def add(self, x: torch.Tensor, metadata: Iterable[str] | None = None) -> None:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        vectors = [memory_pb2.Vector(values=list(map(float, vec.cpu().tolist()))) for vec in x]
        metas = list(metadata) if metadata is not None else []
        request = memory_pb2.AddRequest(vectors=vectors, metadata=metas)
        self.stub.Add(request)

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, List[Any]]:
        vec = memory_pb2.Vector(values=list(map(float, query.cpu().view(-1).tolist())))
        request = memory_pb2.QueryRequest(query=vec, k=k)
        reply = self.stub.Query(request)
        if not reply.vectors:
            return torch.empty(0, query.size(-1)), []
        out = torch.tensor([v.values for v in reply.vectors], dtype=torch.float32)
        return out, list(reply.metadata)
