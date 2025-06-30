import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Any, Tuple, List

import grpc
import numpy as np
import torch

from .hierarchical_memory import HierarchicalMemory

_SERVICE = "DistributedMemory"


class _MemoryServicer:
    def __init__(self, memory: HierarchicalMemory) -> None:
        self.memory = memory

    def push(self, request: bytes, context: grpc.ServicerContext) -> bytes:
        vectors, metadata = pickle.loads(request)
        t = torch.from_numpy(np.asarray(vectors, dtype=np.float32))
        self.memory.add(t, metadata=metadata)
        return b""

    def query(self, request: bytes, context: grpc.ServicerContext) -> bytes:
        query_vec, k = pickle.loads(request)
        t = torch.from_numpy(np.asarray(query_vec, dtype=np.float32))
        out, meta = self.memory.search(t, k=int(k))
        return pickle.dumps((out.detach().cpu().numpy(), meta))


def start_server(memory: HierarchicalMemory, address: str = "[::]:50051") -> grpc.Server:
    """Start a gRPC server exposing ``push`` and ``query``."""
    service = _MemoryServicer(memory)
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    handler = grpc.method_handlers_generic_handler(
        _SERVICE,
        {
            "Push": grpc.unary_unary_rpc_method_handler(
                service.push,
                request_deserializer=lambda x: x,
                response_serializer=lambda x: x,
            ),
            "Query": grpc.unary_unary_rpc_method_handler(
                service.query,
                request_deserializer=lambda x: x,
                response_serializer=lambda x: x,
            ),
        },
    )
    server.add_generic_rpc_handlers((handler,))
    server.add_insecure_port(address)
    server.start()
    return server


def push_remote(
    address: str, vectors: np.ndarray, metadata: Iterable[Any] | None = None
) -> None:
    """Push vectors to a remote server."""
    with grpc.insecure_channel(address) as channel:
        stub = channel.unary_unary(
            f"/{_SERVICE}/Push",
            request_serializer=pickle.dumps,
            response_deserializer=lambda x: x,
        )
        stub((np.asarray(vectors, dtype=np.float32), metadata))


def query_remote(
    address: str, query: np.ndarray, k: int = 5
) -> Tuple[torch.Tensor, List[Any]]:
    """Query vectors from a remote server."""
    with grpc.insecure_channel(address) as channel:
        stub = channel.unary_unary(
            f"/{_SERVICE}/Query",
            request_serializer=pickle.dumps,
            response_deserializer=pickle.loads,
        )
        vecs, meta = stub((np.asarray(query, dtype=np.float32), k))
        return torch.from_numpy(vecs), meta

