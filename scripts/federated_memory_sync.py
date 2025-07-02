import argparse
import time
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_service import serve
from asi.federated_memory_exchange import FederatedMemoryExchange


def start_nodes(num: int, dim: int, capacity: int, start_port: int = 50400):
    servers = []
    exchanges = []
    addresses = []
    for i in range(num):
        mem = HierarchicalMemory(dim=dim, compressed_dim=dim // 2, capacity=capacity)
        addr = f"localhost:{start_port + i}"
        server = serve(mem, addr)
        servers.append(server)
        addresses.append(addr)
        exchanges.append(FederatedMemoryExchange(mem))
    for i, ex in enumerate(exchanges):
        ex.peers = [a for j, a in enumerate(addresses) if j != i]
    return servers, exchanges


def benchmark(num_nodes: int, num_vecs: int, dim: int) -> None:
    servers, exchanges = start_nodes(num_nodes, dim, num_vecs * 2)
    data = torch.randn(num_vecs, dim)
    start = time.perf_counter()
    for v in data:
        exchanges[0].push(v.unsqueeze(0))
    add_t = time.perf_counter() - start

    start = time.perf_counter()
    for ex in exchanges:
        ex.query(data[0], k=1)
    query_t = time.perf_counter() - start

    for s in servers:
        s.stop(0)

    print(f"Replicated {num_vecs} vectors across {num_nodes} nodes in {add_t:.2f}s")
    print(f"Query time per node: {query_t/num_nodes:.4f}s")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Benchmark FederatedMemoryExchange")
    parser.add_argument("--nodes", type=int, default=2, help="Number of memory nodes")
    parser.add_argument("--vectors", type=int, default=100, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=64, help="Vector dimension")
    args = parser.parse_args()
    benchmark(args.nodes, args.vectors, args.dim)
