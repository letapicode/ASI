# Quantized Vector Search Tasks

1. **Analyze scaling requirements**
   - Estimate embedding storage for 200M LOC using existing `PQVectorStore` design.
   - Determine target compression ratio and latency based on 100M LOC results in `docs/Plan.md`.
2. **Extend `PQVectorStore` for sharded indexing**
   - Allow building multiple FAISS `IndexIVFPQ` shards and merging them on demand.
   - Expose `add_shard()` and `search_shards()` helpers.
3. **Build `CodeEmbeddingExtractor`**
   - Parse repositories and embed functions or code blocks with a deterministic model.
   - Store embeddings and metadata in shard files for indexing.
4. **Implement `SnapshotIndexManager`**
   - Track file changes and rebuild affected shards asynchronously.
   - Maintain snapshot versions for accurate search results.
5. **Create `QuantizedSearchService`**
   - Wrap the extended PQ store behind a gRPC API similar to `quantum_memory_server`.
   - Include automatic fallback to uncompressed search when no shard matches.
6. **Design incremental update pipeline**
   - Watch repository diffs and enqueue re-embedding tasks.
   - Support hot-swapping updated shards without downtime.
7. **Benchmark retrieval accuracy and latency**
   - Write `scripts/benchmark_quant_search.py` to compare against `FaissVectorStore`.
   - Target <200ms query time with ≥99.9% recall on typical queries.
8. **Integrate with `HierarchicalMemory`**
   - Add a `use_quantized` option that loads the PQ-based service when available.
   - Ensure existing tests cover the new branch.
9. **Document the workflow**
   - Summarize algorithm choices and tuning guidelines in `docs/Plan.md`.
   - Include instructions for building indexes and running benchmarks.

