# Development Steps Summary

## PR 1
- Added a bullet in `AGENTS.md` requiring each pull request to summarize what changed, how it was done, and why, recorded in `steps_summary.md`.
- Generated `parallel_tasks.md` enumerating a review task for every file in the repository. Each task instructs contributors to analyze code quality, docs, and tests.
- Created this `steps_summary.md` to document the steps themselves.
- Added `ideas.md` to capture reasoning behind feature additions and their connection to ASI goals.

## PR 2
- Refined the bullet in `AGENTS.md` describing the PR summary requirement for brevity.
- Updated Task 1 in `parallel_tasks.md` with concrete review notes for `AGENTS.md`.
- Documented these adjustments in this file.

## PR 3
- Introduced `graph_visualizer_base.py` with helper functions for reading graph JSON, layout calculations and a reusable `WebSocketServer`.
- Refactored `got_visualizer.py`, `got_3d_visualizer.py`, and `ar_got_overlay.py` to use the new helpers, eliminating duplicated code and adding fallback imports for tests.
- Updated `docs/Plan.md` with a bullet about the shared base module.

## PR 4
- Created `cross_lingual_utils.embed_text` as shared deterministic text embedding helper.
- Updated `cross_lingual_memory.py` and `cross_lingual_graph.py` to import this function instead of local implementations.
- Adjusted tests to rely on the new module.

## PR 5
- Introduced `hpc_base_scheduler.HPCBaseScheduler` to handle job queueing and submission.
- Added `ArimaStrategy` and `GNNStrategy` as pluggable forecast components used by the base class.
- Refactored `hpc_forecast_scheduler.py` and `hpc_gnn_scheduler.py` to inherit from the base scheduler.
- Simplified `hpc_multi_scheduler.py` by calling `forecast_scores()` directly on each scheduler instance.
- Documented the new architecture in `docs/Plan.md`.
- Refactored summarizing memories to share BaseSummarizingMemory.

## PR 6
- Documented the quantized search pipeline in `docs/quantized_vector_search_tasks.md`.
- Added `CodeIndexer` and `IncrementalPQIndexer` to embed code and manage PQ shards.
- Extended `HierarchicalMemory` with a `pq_store` and re-ranking search logic.
- Implemented `QuantizedMemoryServer` and client for remote queries.
- Created `build_pq_index.py` script and new unit test `test_quantized_search.py`.
- Updated `Implementation.md`, `Plan.md` and README with usage instructions.

## PR 7
- Fixed failing quantized search test by adding missing `_DummyTensor` helpers
  and tweaking retrieval logic.

## PR 8
- Optimized candidate lookup in `HierarchicalMemory.search` by indexing
  metadata to vector indices.
- Extended `VectorStore`, `FaissVectorStore` and `PQVectorStore` with
  `_meta_map` dictionaries to speed up retrieval.

## PR 9
- Unified carbon-aware scheduling by merging `carbon_hpc_scheduler` into `carbon_aware_scheduler`.
- Added `dashboard_import_helper.load_base_dashboard` and updated key dashboards to use it.
- Documented scheduler changes in `docs/Plan.md`.

## PR 10
- Introduced `_record_carbon_saving` helper used by `submit_best` and
  `submit_best_rl` to consolidate telemetry logic.
- Updated unit tests and documentation accordingly.

## PR 11
- Moved `BaseSummarizingMemory` into `summarizing_memory.py` and removed the old module.
- Updated dependent modules and tests to import `BaseSummarizingMemory` from `asi.summarizing_memory`.
- Adjusted unit tests to load the new unified module with lightweight stubs.

## PR 12
- Added `memory_client_base.MemoryClientBase` with reusable `add_batch` and
  `query_batch` methods shared by gRPC memory clients.
- Refactored `RemoteMemory` and `QuantizedMemoryClient` to inherit the base and
  renamed `RemoteMemory.search_batch` to `query_batch`.
- Updated unit tests and exported the base class. Documented the change in
  `docs/Plan.md`.

## PR 13
- Removed `src/hpc_scheduler.py` in favour of `asi.hpc_schedulers`.
- Updated all scheduler modules, tests and documentation to import from the new
  package path.
- Confirmed scheduler-related tests run with pytest.

## PR 14
- Replaced manual BaseDashboard fallbacks in several dashboards with
  `load_base_dashboard` from `dashboard_import_helper`.
- Added helper import fallbacks and cleaned up unused imports.
- Updated dashboard tests with lightweight stubs for missing dependencies so
  they run without external packages.
