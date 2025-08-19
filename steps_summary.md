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
- Consolidated `data_provenance_ledger.py` and `blockchain_provenance_ledger.py` into a unified `provenance_ledger.py` with a shared base class.
- Updated imports across modules, scripts, docs, and tests to reference the new ledger module.
- Removed redundant ledger files and adjusted documentation to describe the consolidated design.
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
- Consolidated dataset lineage visualization by merging `dataset_lineage_dashboard`
  into `lineage_visualizer` with a shared `_build_graph` helper.
- Updated imports across modules, scripts, tests and documentation to reference
  the unified module.

## PR 14
- Replaced manual BaseDashboard fallbacks in several dashboards with
  `load_base_dashboard` from `dashboard_import_helper`.
- Added helper import fallbacks and cleaned up unused imports.
- Updated dashboard tests with lightweight stubs for missing dependencies so
  they run without external packages.

## PR 15
- Renamed `tests/test_hpc_scheduler.py` to `tests/test_hpc_schedulers.py` for clarity.
- Documented the import path `asi.hpc_schedulers` in `docs/Plan.md`.

## PR 16
- Integrated temperature-aware throttling directly into `AcceleratorScheduler`.
- Removed legacy `gpu_aware_scheduler.py` and `thermal_gpu_scheduler.py` modules.
- Updated tests to use `AcceleratorScheduler(max_util=..., max_temp=...)`.
- Adjusted documentation for the new behaviour.


- Reworked `EphemeralVectorStore` to inherit from `VectorStore` and reuse its
  insertion logic.
- Added timestamp tracking so expired vectors are purged automatically before
  searching, deleting or checking the length.
- Updated unit tests to rely on this implicit cleanup and noted the subclass in
  `docs/Plan.md`.

- Removed the legacy `carbon_hpc_scheduler` module.
- Updated all imports and tests to use `asi.carbon_aware_scheduler`.
- Documented the change and cleaned up related task descriptions.

- Unified forecasting strategies in `forecast_strategies.py`.
- Removed algorithm-specific scheduler modules.
- Added `make_scheduler()` factory in `hpc_base_scheduler`.
- Updated code, tests and scripts to use the new strategy module.
- Documented the changes in `docs/Plan.md`.

## PR 17
- Moved gRPC helper functions (`push_remote`, `query_remote` and batch/async
  variants) from `hierarchical_memory.py` into `remote_memory.py`.
- Updated modules and tests to import the unified helpers.
- Adjusted documentation in `Implementation.md` to reference the new location.

## PR 18
- Extracted `BaseMemoryServer` with common start/stop and push/query logic.
- Updated `MemoryServer` and specialized variants to inherit from it.
- `serve()` now returns a `BaseMemoryServer` instance.
- Documented the change in `docs/Plan.md`.
- Consolidated gRPC memory clients into `memory_clients.py` with
  `RemoteMemoryClient`, `QuantumMemoryClient`, `QuantizedMemoryClient` and
  `EdgeMemoryClient`. Updated modules, tests and docs accordingly.

## PR 19
- Removed `memory_client_base.py`, `edge_memory_client.py`,
  `quantized_memory_client.py`, `quantum_memory_client.py` and
  `remote_memory.py`.
- Updated all imports to reference `memory_clients.py` directly and dropped the
  obsolete `RemoteMemory` alias.
- Documentation now states `memory_clients.py` is the single entry point for all
  gRPC clients.



- Consolidated hardware backends and retention modules into single shared files.
## PR 20
- Consolidated all dashboard modules into `src/dashboards.py`.
- Removed the old individual files and updated imports across the package.
- Updated documentation references in `Implementation.md` and `Plan.md`.
- Adjusted unit tests to load dashboard classes from the new module.

## PR 21
- Combined `ElasticMoERouter` and `RLMoERouter` into `src/moe_router.py`.
- Removed the old router modules and updated imports across code and docs.
- Documented the consolidation in `docs/Implementation.md` and `docs/Plan.md`.
- Updated tests to import from the unified router module.


## PR 22
- Merged all RL scheduler implementations into `src/rl_schedulers.py`.
- Removed obsolete scheduler modules and updated imports across tests, scripts, docs and package `__init__`.
- Added `translator_fallback.py` to share a stub `CrossLingualTranslator`, used by dataset watcher and fairness evaluator.
- Documented the new module references in `docs/Plan.md` and `docs/Implementation.md`.

## PR 23
- Consolidated proof helpers into `src/proofs.py` and removed the old modules.
- Removed `hybrid_retention.py` and `retnet_retention.py`; `src/retention.py` now exports both classes.
- Updated imports across the package and adjusted unit tests.
- Cleaned up documentation references in `docs/Implementation.md` and `docs/Plan.md`.

## PR 24
- Merged `hpc_base_scheduler.py` and `hpc_multi_scheduler.py` into `hpc_schedulers.py`.
- Added `HPCBaseScheduler`, `MultiClusterScheduler` and `make_scheduler` to the unified module.
- Updated all imports, scripts, and tests to reference the new path.
- Documented the consolidated scheduler in `docs/Plan.md` and exported the helpers in `__init__.py`.

## PR 25
- Combined all memory server implementations into `src/memory_servers.py`.
- Old modules now re-export the classes to keep backward compatibility.
- Updated `hierarchical_memory.py` to import the unified `MemoryServer`.
- Documented the change in `docs/Plan.md`.
\n## PR 26
- Unified various schedulers into `schedulers.py` and moved shared hardware checks to `scheduler_utils.py`.
- Old scheduler modules now re-export from the new file to maintain compatibility.
- Updated `__init__.py` imports accordingly.

## PR 27
- Consolidated fairness utilities into `src/fairness.py`.
- Legacy modules now import classes from the unified file for backward compatibility.
- Updated dependent modules, docs, and tests to reference the new path.

## PR 28
- Created `fairness_wrappers.py`, `scheduler_wrappers.py`, and `memory_server_wrappers.py` to consolidate repetitive wrapper logic.
- Updated individual wrapper modules to import from these new files.
- This removes duplicated import boilerplate while keeping backward compatibility.

## PR 29
- Merged `fairness_adaptation.py` into `fairness.py` and re-exported the class via `fairness_wrappers`.
- Updated imports in `__init__.py` and tests to reference the unified module.
- Documented the consolidation in `docs/Plan.md`.

## PR 30
- Added `torch_fallback.py` providing `DummyTensor` and lightweight `torch`/`nn` replacements.
- Removed duplicate fallback classes from several modules and imported the shared helper instead.
- Unified deterministic text embedding logic by using `cross_lingual_utils.embed_text` across modules.
- Refactored `analogical_retrieval.py` to rely on the shared embedding helper.

## PR 31
- Removed legacy wrapper modules for memory servers, schedulers and fairness utilities.
- Updated imports across code and tests to load classes from `memory_servers.py`, `schedulers.py` and `fairness.py` directly.
- Adjusted `memory_service.serve` and `distributed_trainer` to use the unified modules.
- Documented the cleanup in `docs/Plan.md`.

## PR 32
- Consolidated dataset lineage client, server and manager into `dataset_lineage.py`.
- Updated imports across source, tests, scripts and docs to use the unified module.
- Revised documentation and task references for the new dataset lineage module.

## PR 33
- Merged `BudgetAwareScheduler` into `schedulers.py` and removed the standalone module.
- Moved `AdaptiveCostScheduler` and price-aware helpers into `rl_schedulers.py` and `carbon_aware_scheduler.py`.
- Updated imports, tests, scripts and documentation to reference the unified scheduler modules.

## PR 34
- Combined `quantum_retrieval.py` and `quantum_sampler.py` into `quantum_sampling.py` with a shared softmax helper.
- Updated source modules, tests, and package exports to import from the new consolidated module.
- Revised documentation and task references to reflect the unified quantum sampling utilities.
- Corrected import statements in `quantum_sampling.py`.

## PR 35
- Merged `graph_visualizer_base.py`, `got_visualizer.py` and
  `got_3d_visualizer.py` into a unified `graph_visualizer.py`.
- Updated imports across modules, scripts and tests to reference the
  consolidated visualizer.
- Documented the new module in `docs/Plan.md` and revised related review tasks.

## PR 36
- Merged `scripts/code_refine.py` and `scripts/secure_dataset_exchange.py` into their corresponding modules under `src/`.
- Added CLI entry points to `code_refine.py` and `secure_dataset_exchange.py` and removed the duplicate scripts.
- Updated tests, documentation and task references to use `python -m asi.code_refine` and `python -m asi.secure_dataset_exchange`.

## PR 37
- Consolidated dataset bias utilities by folding `DataBiasMitigator` into `dataset_bias_detector.py`.
- Removed the standalone `data_bias_mitigator.py` and updated imports across the package and tests.
- Documented the unified bias module in `docs/Plan.md`.

## PR 38
- Merged `lineage_visualizer.py` and `kg_visualizer.py` into a unified `graph_visualizers.py` with shared `D3GraphVisualizer` base.
- Updated imports across modules, scripts, tests, and documentation to reference the consolidated visualizers.

## PR 39
- Consolidated all graph visualizers into `graph_visualizer.py`.
- Removed the old `graph_visualizers.py` module and redirected imports in code, tests, scripts and documentation.

## PR 40
- Unified pull request utilities by merging `pull_request_monitor` and
  `pr_conflict_checker` into `pull_request_tools`.
- Updated imports, tests, and documentation to use the consolidated module and
  its CLI subcommands.

## PR 41
- Consolidated privacy utilities into a single `privacy.py` module housing `PrivacyGuard`,
  `PrivacyBudgetManager`, and `PrivacyAuditor`.
- Removed the old dedicated modules and updated all imports across the codebase,
  scripts, tests, and documentation to reference the unified file.
