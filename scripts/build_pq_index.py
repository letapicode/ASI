#!/usr/bin/env python
"""Build a PQ index for source code."""

from pathlib import Path
import argparse

from asi.code_indexer import CodeIndexer
from asi.incremental_pq_indexer import IncrementalPQIndexer


def main() -> None:
    p = argparse.ArgumentParser(description="Build PQ index")
    p.add_argument("source", help="source directory")
    p.add_argument("out", help="output directory")
    p.add_argument("--dim", type=int, default=32)
    args = p.parse_args()

    indexer = CodeIndexer(args.source, dim=args.dim)
    pq = IncrementalPQIndexer(args.dim, args.out)
    for vec, meta in indexer.index():
        pq.add(vec.reshape(1, -1), [meta])
    pq.save()
    indexer.save()


if __name__ == "__main__":
    main()
