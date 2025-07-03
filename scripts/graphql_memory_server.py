import argparse
import time
import torch
from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_service import serve
from asi.graphql_memory_gateway import GraphQLMemoryGateway

def benchmark(queries: int, dim: int) -> None:
    mem = HierarchicalMemory(dim=dim, compressed_dim=dim // 2, capacity=queries * 2)
    server = serve(mem, "localhost:50555")
    gateway = GraphQLMemoryGateway("localhost:50555")
    vecs = torch.randn(queries, dim)
    for v in vecs:
        mem.add(v.unsqueeze(0))
    start = time.perf_counter()
    for v in vecs:
        qstr = "{ query(vector: [%s], k: 1) }" % ",".join(f"{x:.4f}" for x in v.tolist())
        gateway.execute(qstr)
    elapsed = time.perf_counter() - start
    server.stop(0)
    print(f"{queries} queries in {elapsed:.3f}s -> {queries/elapsed:.2f}/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GraphQL memory gateway")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--dim", type=int, default=16)
    args = parser.parse_args()
    benchmark(args.queries, args.dim)
