# Quantized Vector Search Tasks

This document outlines the planned pipeline for scalable code search using product quantization.
The goal is to store hundreds of millions of code embeddings while keeping recall and latency
close to an exact FAISS index.

## Approach

1. Extract deterministic line embeddings from source files.
2. Build a sharded `PQVectorStore` and update it incrementally when code changes.
3. Query the quantized index for candidates and re-rank them with full-precision vectors.
4. Serve the combined index through a lightweight gRPC service.

## Implementation Tasks

1. **Incremental code embedding pipeline** – `code_indexer.py` streams files,
   tokenizes each line and re-embeds only changed content.
2. **Large-scale PQ index management** – `incremental_pq_indexer.py` trains a
   product quantizer on a sample of the codebase and maintains disk-backed shards.
3. **Candidate search + re-ranking** – `HierarchicalMemory.search()` queries the
   quantized shards first and re-ranks results using the full vectors.
4. **Remote query service** – `QuantizedMemoryServer` exposes batched push and
   query RPCs with a thin client.
5. **Index build script** – `scripts/build_pq_index.py` runs the indexer and
   saves the shards for serving.
6. **Accuracy and latency tests** – `tests/test_quantized_search.py` verifies
   ≥99.9 % recall and benchmarks latency.
7. **Documentation and README** – instructions for building and using the
   quantized search pipeline.
