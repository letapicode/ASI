import argparse
import time
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.distributed_memory import DistributedMemory
from asi.memory_service import serve


def start_servers(num, dim, compressed_dim, capacity, port_start=50300):
    servers = []
    addresses = []
    for i in range(num):
        mem = HierarchicalMemory(dim=dim, compressed_dim=compressed_dim, capacity=capacity)
        addr = f"localhost:{port_start + i}"
        server = serve(mem, addr)
        servers.append(server)
        addresses.append(addr)
    return servers, addresses


def benchmark(memory, num_vecs, dim):
    data = torch.randn(num_vecs, dim)
    start = time.perf_counter()
    for v in data:
        memory.add(v.unsqueeze(0))
    add_time = time.perf_counter() - start

    start = time.perf_counter()
    for v in data:
        memory.search(v, k=1)
    query_time = time.perf_counter() - start
    return num_vecs / add_time, num_vecs / query_time


def run(num_servers, num_vecs, dim):
    if num_servers > 0:
        servers, addrs = start_servers(num_servers, dim, dim // 2, num_vecs * 2)
        mem = DistributedMemory(dim=dim, compressed_dim=dim // 2, capacity=num_vecs * 2, remotes=addrs)
    else:
        servers = []
        mem = HierarchicalMemory(dim=dim, compressed_dim=dim // 2, capacity=num_vecs * 2)
    add_t, query_t = benchmark(mem, num_vecs, dim)
    for s in servers:
        s.stop(0)
    return add_t, query_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DistributedMemory throughput")
    parser.add_argument("--servers", type=int, default=4, help="Number of remote memory servers")
    parser.add_argument("--vectors", type=int, default=100, help="Number of vectors to add/query")
    parser.add_argument("--dim", type=int, default=64, help="Dimension of each vector")
    args = parser.parse_args()

    single_add, single_query = run(0, args.vectors, args.dim)
    dist_add, dist_query = run(args.servers, args.vectors, args.dim)

    print(f"Single-node add throughput: {single_add:.2f}/s | query: {single_query:.2f}/s")
    print(
        f"{args.servers}-node distributed add throughput: {dist_add:.2f}/s | query: {dist_query:.2f}/s"
    )
