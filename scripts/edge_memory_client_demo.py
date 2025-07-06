import argparse
import time
import torch

from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_service import serve
from asi.edge_memory_client import EdgeMemoryClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo EdgeMemoryClient")
    parser.add_argument(
        "--offline", action="store_true", help="start client offline then sync"
    )
    args = parser.parse_args()

    dim = 4
    mem = HierarchicalMemory(dim=dim, compressed_dim=2, capacity=10)
    server = None
    if not args.offline:
        server = serve(mem, "localhost:50510")

    client = EdgeMemoryClient("localhost:50510", buffer_size=1, sync_interval=0.5)

    data = torch.randn(3, dim)
    for i, vec in enumerate(data):
        client.add(vec, metadata=[f"m{i}"])
        time.sleep(0.1)

    if args.offline:
        server = serve(mem, "localhost:50510")
        time.sleep(1.0)

    out, meta = client.search(data[0], k=1)
    print("retrieved", out.shape, meta)
    client.close()
    if server:
        server.stop(0)


if __name__ == "__main__":  # pragma: no cover
    main()
