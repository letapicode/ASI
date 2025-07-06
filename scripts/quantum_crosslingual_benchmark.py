import argparse
import time
import numpy as np

from asi.cross_lingual_memory import CrossLingualMemory
from asi.data_ingest import CrossLingualTranslator


def run_benchmark(samples: int = 200, dim: int = 16, k: int = 1) -> None:
    rng = np.random.default_rng(0)
    translator = CrossLingualTranslator(["es"])
    mem = CrossLingualMemory(
        dim=dim,
        compressed_dim=dim // 2,
        capacity=samples * 2,
        translator=translator,
    )
    data = rng.normal(size=(samples, dim)).astype(np.float32)
    mem.add(data, metadata=list(range(samples)))
    queries = data[:20] + rng.normal(scale=0.01, size=(20, dim)).astype(np.float32)

    start = time.perf_counter()
    classical_hits = 0
    for i, q in enumerate(queries):
        _, meta = mem.search(q, k=k)
        if meta and meta[0] == i:
            classical_hits += 1
    classical_time = time.perf_counter() - start

    start = time.perf_counter()
    quantum_hits = 0
    for i, q in enumerate(queries):
        _, meta = mem.search(q, k=k, quantum=True)
        if meta and meta[0] == i:
            quantum_hits += 1
    quantum_time = time.perf_counter() - start

    print(
        f"classical_hits: {classical_hits} time: {classical_time:.4f}s\n"
        f"quantum_hits:   {quantum_hits} time: {quantum_time:.4f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantum cross-lingual retrieval benchmark"
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("-k", type=int, default=1)
    args = parser.parse_args()
    run_benchmark(args.samples, args.dim, args.k)
