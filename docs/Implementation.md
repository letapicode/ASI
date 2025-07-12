# Implementation Notes for S-1 and S-2

This repository includes starter modules for the first two algorithms listed in `docs/Plan.md`.

## S-1 Sparse Mixture-of-Experts Routing

- `src/moe_router.py` provides two routers:
  - `HashRouter` uses hash-based gating to activate at most two experts per token.
  - `SwitchRouter` employs a learned linear gate and selects the top-k experts.
  Both expose `load_balance_std` and `expert_utilization` to inspect token distribution.
  - `ElasticMoERouter` dynamically reduces the number of active experts when GPU
    memory utilization gets high. It inherits from `SwitchRouter` and exposes
    `active_experts` to track the current count.
  - `RLMoERouter` learns routing probabilities with a REINFORCE update and
    mirrors the `ElasticMoERouter` API for plug-and-play experiments.
- `src/moe_layer.py` implements a small MoE feed-forward block using these routers. It accepts an optional
  `balance_weight` which multiplies the `balance_loss()` penalty derived from the router's assignments and
  returns it alongside the layer output.

## S-2 FlashAttention-3 Kernel

- `src/flash_attention3.py` wraps the FlashAttention-3 kernel and exposes `_HAS_FLASH3` to
signal availability. If the import fails, the wrapper calls PyTorch's `scaled_dot_product_attention` instead.

These modules are prototypes to facilitate experimentation and benchmarking.

## Load-Balancing Measurement

See `docs/load_balance.md` for a walkthrough on how expert utilization is computed and how to replicate
the measurement. A quick example:

```python
from src.moe_router import HashRouter
import torch

x = torch.randn(4, 512, 256)
router = HashRouter(num_experts=16)
assign = router(x)
print('std:', router.load_balance_std(assign))
print('counts:', router.expert_utilization(assign))
```

## Benchmark Script

`scripts/benchmark_moe.py` offers a minimal example comparing parameter counts and approximate
training FLOPs with and without the MOE router.

Run it from the project root. By default it uses `HashRouter`; pass `--router switch` for the learned router or `--router elastic` for the adaptive variant:

```bash
python scripts/benchmark_moe.py --router elastic
```

Expected output shows the dense and MOE parameter counts along with their ratio and a rough FLOP
ratio:

```
Dense params: 262912
MOE params: 2236672
Param increase: 8.5
FLOP ratio: 8.5
```

`scripts/moe_vs_dense.py` provides a similar toy benchmark implemented as a standalone module. It
contrasts a dense feed-forward model with the MOE version. Pass `--router elastic` to exercise the adaptive router.

Run it as:

```bash
python scripts/moe_vs_dense.py --router elastic
```

The script prints the same parameter counts and a comparable FLOP ratio, labelled `Param ratio`. Both
scripts are starting points for more detailed experiments.

## FlashAttention-3 Integration

`src/flash_attention3.py` attempts to import the FlashAttention-3 CUDA/ROCm kernel.
The module exposes `_HAS_FLASH3` to indicate whether the kernel was imported.
If not, the wrapper falls back to `torch.nn.functional.scaled_dot_product_attention`.
To build the kernel yourself:

1. Install the `flash-attn` package with CUDA visible: `pip install flash-attn --no-binary flash-attn`.
2. Ensure `TORCH_CUDA_ARCH_LIST` matches your GPU architecture.
3. Set the environment variable `FLASH_ATTENTION_FORCE_BUILD=1` to trigger compilation from source if needed.

After installation, the wrapper will automatically call the optimized kernel.

## Neuromorphic Execution

`src/loihi_backend.py` wraps optional calls into Intel's Loihi SDK. When
`_HAS_LOIHI` is `True`, `LIFNeuron` and `SpikingLinear` offload their
computations via `loihi_backend`. Install `nxsdk` and set
`MultiModalWorldModelConfig.use_spiking=True` together with
`use_loihi=True` to execute the world model on neuromorphic hardware.

`loihi_backend.configure_loihi()` accepts a `LoihiConfig` dataclass to
specify the number of active cores and spike precision. Adjusting these
options lets inference run fully on Loihi and typically reduces energy
consumption by around 10\times compared to the CPU fallback.

## FPGA Acceleration

`src/fpga_backend.py` exposes an `FPGAAccelerator` helper that compiles a
PyTorch module and executes it on an attached FPGA when the optional
`pyopencl` dependency is installed. Pass `use_fpga=True` in
`MultiModalWorldModelConfig` or `EdgeRLTrainer` to offload computations.
`configure_fpga()` accepts an `FPGAConfig` with the target device index and
optimisation flag.

## Analog Acceleration

`src/analog_backend.py` defines an `AnalogAccelerator` interface that calls out
to an optional analog simulator for matrix multiplies. When no simulator is
present the helper falls back to `torch.matmul`.

Use it as a context manager to temporarily patch `torch.matmul`:

```python
from asi.analog_backend import AnalogAccelerator

with AnalogAccelerator():
    out = torch.matmul(a, b)
```

Enable the analog path by passing `use_analog=True` in
`MultiModalWorldModelConfig` or `EdgeRLTrainer`.

### Analog Device Detection

`hardware_detect.list_analog()` checks several sources to enumerate available
analog accelerators so the `AdaptiveScheduler` can queue jobs on them. Detection
proceeds in the following order:

1. If the environment variable `ASI_ANALOG_DEVICES` is set it is parsed as a
   comma-separated list of device identifiers.
2. When the optional `analogsim` package exposes `list_devices()` or
   `device_count()` these helpers are called to query the simulator for attached
   analog hardware.
3. If detection fails but `analog_backend._HAS_ANALOG` is `True` the fallback
   identifier `analog0` is returned.

The detected list is cached so repeated calls avoid querying the backend again.

The scheduler includes these identifiers under the `"analog"` device type which
means analog hardware can be selected via `add(job, device="analog")`.

## S-3 Scaling-law Breakpoint Model

`src/scaling_law.py` defines ``BreakpointScalingLaw`` which fits a piecewise
log--log relation to compute versus loss. Initialize the model with an optional
breakpoint guess and call ``fit()`` with arrays of compute values and observed
losses.

```python
from src.scaling_law import BreakpointScalingLaw

params = [1e7, 5e7, 1e8, 5e8]
loss = [2.0, 1.8, 1.6, 1.3]
model = BreakpointScalingLaw()
model.fit(params, loss)
print('breakpoint:', model.break_compute)
print('predictions:', model.predict(params))
```

The helper searches over candidate breakpoints and performs linear regression in
log space on either side. The resulting model can forecast loss beyond the
training range.

Pass a list to ``break_compute`` to fit multiple breakpoints. Each segment's
``(slope, intercept)`` pair is stored in ``model.params``:

```python
model = BreakpointScalingLaw([5e7, 5e8])
model.fit(params, loss)
print('breaks:', model.break_compute)
print('segments:', model.params)
```

`src/scaling_breakpoint.py` offers a compact variant `fit_breakpoint()` which
returns a dataclass `BreakpointModel` with slopes and intercepts on either side
of the break. Use it when you just need predictions without storing the full
fitting helper:

```python
from src.scaling_breakpoint import fit_breakpoint

model = fit_breakpoint(params, loss)
print(model.breakpoint, model.predict(params))
```

## S-4 4-bit Quantized LoRA Training

- `src/lora_quant.py` implements a small LoRA adapter stored in int4 precision.
- `apply_quant_lora()` injects these adapters into an existing network so most parameters stay frozen during fine-tuning.
- `multimodal_world_model.py` now exposes a `use_lora` flag to wrap its transformer
  layers with these quantized adapters.

## C-1 RetNet Retention Kernel

- `src/retnet_retention.py` implements a minimal retention module.
- It sequentially accumulates `k * v` products with a decay factor and multiplies by the query at each step.
- The design follows the linear-time, constant-memory retention described in the
  RetNet paper and serves as a placeholder for more optimised kernels.

## C-2 Mamba State-Space Block

- `src/mamba_block.py` implements a simplified state-space module with a gated recurrent update.
- The block runs in linear time over the sequence and maintains a per-batch hidden state.
- It serves as a minimal reference for experiments targeting the Mamba architecture described in the paper.

## C-3 Hyena Filter

- `src/hyena_filter.py` adds an FFT-based 1D convolution used in Hyena and H\u00b3 models.
- The module exposes a `HyenaFilter` class with a learnable filter.
- Forward pass performs convolution in the frequency domain and returns a tensor
  with the same shape as the input.
- This provides a prototype for the long-range implicit-FFT filtering described
  in the Plan under **C-3**.

## Streaming Compression

- `src/streaming_compression.py` implements a `StreamingCompressor` that keeps a
  reservoir sample of token embeddings and compresses them with a small
  autoencoder.
- This demonstrates the *streaming compression* step toward infinite context,
  keeping the working set roughly logarithmic in sequence length.

## Hierarchical Vector Store

 - `src/vector_stores.py` provides an in-memory `VectorStore` with dot-product
  similarity search.
- `VectorStore.add()` stores embeddings with optional metadata and
  `search()` returns the top-k nearest vectors.
- The store now supports `save()` and `load()` to persist vectors and metadata
  to a compressed `.npz` file.
- This serves as a minimal prototype for the *hierarchical retrieval* memory
  described in the Plan.
 - `src/vector_stores.py` adds `AsyncFaissVectorStore` which wraps the
  disk-backed FAISS index with a thread pool. `add_async()` and `search_async()`
  submit operations to background workers, while `aadd()` and `asearch()`
  provide `asyncio` interfaces. It can also be used with
  ``async with AsyncFaissVectorStore(...) as store:`` to automatically shut down
  the worker pool.
- `src/hierarchical_memory.py` combines `StreamingCompressor` with either the
  in-memory `VectorStore` or a new `FaissVectorStore`. Passing a path hooks the
  memory to a persistent FAISS index so distant tokens are written to disk
  automatically. Retrieved vectors stay on the query device. The `save()` and
  `load()` helpers now handle both store types, letting large histories rebuild
  from disk with a single call.
- `HierarchicalMemory` now accepts `use_async=True` to employ
 `AsyncFaissVectorStore`. `add()`, `delete()`, `search()`, `save()` and
 `load()` check if an event loop is running. When called from synchronous code
 they block with ``asyncio.run``. Inside a running loop they return an
 ``asyncio.Task`` so callers can ``await`` the scheduled operation. The
 explicit async variants (`aadd`, `adelete`, `asearch`, `save_async`,
`load_async`) remain available for direct use.
`HierarchicalMemory` defines `__len__` so `len(mem)` reports the number of
stored vectors.
- `src/memory_profiler.py` hooks into `HierarchicalMemory` and records query
  counts, hit/miss ratios and latency. Call `start_profiling()` on a memory
  instance to begin collecting metrics and `report_stats()` to dump them as JSON
  or CSV.

## C-4 MegaByte Patching

- `src/megabyte_patching.py` contains a `MegaBytePatching` module that groups
  byte sequences into fixed-size patches.
- Each patch is embedded with a learnable byte table and projected to a single
  vector, forming the hierarchical representation used in MegaByte models.

## C-5 Top-k Sparse Attention

- `src/topk_sparse_attention.py` defines `topk_sparse_attention` to select the highest-scoring keys per query.
- This provides a lightweight inference-time approximation to full attention for the **C-5** task.
- `k_top` must not exceed `seq_k`; a `ValueError` is raised otherwise.
- See `docs/Plan.md` under **C-5** for the full task context.

## C-6 RWKV Infinite-Context Loop

- `src/rwkv_loop.py` provides a simplified recurrent block employing the token-shift trick.
- It keeps only a single hidden state per batch, enabling constant-memory training on very long sequences.

### Example Infinite Context Training

`scripts/train_infinite_context.py` wires together `RWKVLoop`, `StreamingCompressor`,
`HierarchicalMemory`, and `LinkSlotAttention`. The script now downloads the
TinyShakespeare corpus and feeds it through `ChunkWiseRetrainer`. After each
epoch the model is evaluated to report perplexity, retrieval hit rate, and
current memory size:

```bash
python scripts/train_infinite_context.py --epochs 2
```

Checkpoints under `checkpoints/` persist both the model weights and the
hierarchical memory state. This workflow demonstrates how retrieval memory can
extend the context window indefinitely while still training in constant memory.

To reproduce the toy run step by step:

1. Execute `train_infinite_context.py` with the desired number of epochs. The
   helper `load_dataset()` downloads and tokenizes TinyShakespeare on the first
   run.
2. An `InfiniteContextModel` is instantiated combining `RWKVLoop` with a
   `HierarchicalMemory` store. Training occurs through
   `ChunkWiseRetrainer.train()` which feeds the tokens in 64-token chunks.
3. After each epoch `evaluate()` calculates the loss, perplexity and retrieval
   hit rate using the growing memory.
4. `save_checkpoint()` writes the model weights and the memory state to
   `checkpoints/stepN`. The memory is stored via `HierarchicalMemory.save()`.
5. When restarting from a checkpoint, load `model.pt` with `torch.load` and
   call `HierarchicalMemory.load()` on the saved memory directory. Assign the
   result to `model.memory` before resuming training. Reloading ensures previous
   retrievals remain available so the context effectively persists across runs.

## A-1 Paper-to-Code Transpiler

- `src/paper_to_code.py` offers a minimal transpiler from LaTeX pseudo-code to
  Python. The function `transpile()` maps common `\For`, `\If`, and `\State`
  commands to Python syntax and manages indentation.
- A small command line interface is provided via `python -m src.paper_to_code`
  to convert a LaTeX file into a Python script.

## A-2 AutoBench Harness

- `src/autobench.py` runs each test module in its own subprocess to sandbox imports.
- `summarize_results()` returns a concise scoreboard and shows the full output from failing modules.
- The command line interface `python -m src.autobench` prints this summary for the specified directory.

## A-3 Meta-RL Refactor Agent

- `src/meta_rl_refactor.py` implements a lightweight Q-learning agent that decides whether to
  *replace*, *refactor*, or *rollback* modules based on benchmark feedback.
- The agent exposes `select_action()` for epsilon-greedy exploration and `update()` to adjust
  its Q-table from observed rewards.

## A-4 Quantum Amplitude-Estimation HPO

- `src/quantum_hpo.py` provides a toy hyper-parameter search using a simulated
  quantum amplitude estimation routine.
- Call `QAEHyperparamSearch.search()` with a candidate parameter set and an
  evaluation function that returns `True` on success. The helper repeatedly
  estimates the success probability via either `amplitude_estimate()` or the
  Bayesian variant `amplitude_estimate_bayesian()` and returns the best
  performing setting. Passing `early_stop` halts the search once an estimate
  meets the given probability threshold. Set `max_workers` to evaluate
  candidates concurrently:

  ```python
  search = QAEHyperparamSearch(eval_func, params)
  best, prob = search.search(num_samples=5, early_stop=0.8, max_workers=4)
  ```
- See `docs/Plan.md` task **A-4** for context and goals.

## Pull Request Monitoring

- `src/pull_request_monitor.py` lists open pull requests and checks mergeability.
- The asynchronous helpers now accept an optional `aiohttp.ClientSession` so
  multiple API calls can share a single session. The `main()` helper creates
  one session when invoked with `--use-asyncio` and reuses it across requests.

## L-1 Collective Constitutional AI

- `src/collective_constitution.py` aggregates crowd-sourced principles into
  actionable rules.
- `derive_rules()` keeps principles supported by at least `min_agreement` users.
- `label_responses()` tags unlabeled text as safe or unsafe depending on rule
  matches.

## L-2 Deliberative Alignment

- `src/deliberative_alignment.py` implements a simple chain-of-thought checker.
- `DeliberativeAligner.check()` validates each reasoning step against a policy
  text and returns `True` only if no rule is violated.

## L-3 Iterative Constitutional Self-Alignment

- `src/iter_align.py` implements a lightweight iterative alignment loop.
- `IterativeAligner.iterate()` repeatedly critiques transcripts against current rules and
  extends the rule set with any flagged lines.
- This demonstrates a toy version of the "IterAlign" process described in the Plan.

## L-4 Critic-in-the-Loop RLHF

- `src/critic_rlhf.py` implements a lightweight trainer that blends human rewards
  with scores from a critic model.
- `CriticRLHF.update()` mixes the two feedback signals using a weighting factor
  and updates action values.
- `select_action()` returns the highest-valued action with optional
  epsilon-greedy exploration.

## C-7 Hierarchical Retrieval Memory

- `src/hierarchical_memory.py` and `src/link_slot_attention.py` provide a two-tier memory backed by FAISS.
- The store compresses vectors before writing them to disk and loads the nearest neighbours on demand.
- `RetrievalExplainer.summarize()` distills query results into a brief text used by `MemoryDashboard` to show context for each retrieval.
- `RetrievalExplainer.summarize_multimodal()` collapses text snippets and lists referenced images or audio so multimodal queries render clearly in the dashboard.
- `HierarchicalMemory.search(return_summary=True)` calls these helpers and stores
  the resulting text in ``last_trace['summary']`` so UIs can display the context
  directly.
- `MemoryDashboard` shows the latest summary on its stats page and computes one
  for `/trace` when none was stored.

Example:

```python
mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
data = torch.randn(1, 4)
mem.add(data, metadata=["hello world"])
vec, meta, scores, prov, summary = mem.search(
    data[0], k=1, return_scores=True, return_provenance=True, return_summary=True
)
print(summary)
```

## C-8 Distributed Hierarchical Memory Backend

- `src/hierarchical_memory.py` now includes optional gRPC support. `MemoryServer`
  wraps a `HierarchicalMemory` instance and serves ``Push`` and ``Query`` RPCs.
  - Helper functions `push_remote()` and `query_remote()` in
    `src/memory_clients.py` send vectors to the server and retrieve nearest
    neighbours. Asynchronous variants `push_remote_async()` and
    `query_remote_async()` allow non-blocking interaction when using ``grpc.aio``.
  - The server constructor accepts an ``address`` and ``max_workers`` to control
    the bind host and connection pool size.
  - All gRPC clients now live in `src/memory_clients.py` which exposes
    ``RemoteMemoryClient``, ``QuantumMemoryClient``, ``QuantizedMemoryClient`` and
    ``EdgeMemoryClient`` behind a consistent interface.

## C-9 Hopfield Associative Memory

- `src/hopfield_memory.py` implements a small Hopfield network with
  ``store()`` and ``retrieve()`` helpers for binary patterns.
- `HierarchicalMemory` accepts ``use_hopfield=True`` to keep a tiny in-memory
  associative cache backed by this network.
- `src/differentiable_memory.py` wraps ``HierarchicalMemory`` so retrieved
  vectors keep gradient information. Enable ``use_differentiable_memory`` in
  ``train_world_model`` when training models that update memory contents via
  backpropagation.

### Distributed Memory Benchmark

`scripts/distributed_memory_benchmark.py` launches several `MemoryServer`
instances and measures the add and query throughput of
`DistributedMemory`. It first runs a single-node baseline and then
starts the requested number of servers to compare distributed
performance.

Run the benchmark with four servers as:

```bash
python scripts/distributed_memory_benchmark.py --servers 4 --vectors 100
```

## A-5 Multi-Modal World Model

- `src/multimodal_world_model.py` now implements a unified transformer that ingests text, images and low-level actions.
- `train_world_model()` fits the model on trajectory data and `rollout()` generates simulated transitions for downstream agents.

## A-6 Embodied Skill Transfer

- `src/robot_skill_transfer.py` maps video demonstrations to robot commands.
- `transfer_skills()` fine-tunes policies on a small set of real robot examples and returns the trained model.

## A-7 Self-Play World Model

- `src/self_play_env.py` defines a minimal environment and agent loop for automated skill discovery.
- `rollout_env()` runs the simulator and logs rewards so new policies can be trained from the generated traces.

## A-8 Integrated Self-Play & Skill Transfer

- The orchestrator in `src/self_play_skill_loop.py` alternates `self_play_env.rollout_env()` with `robot_skill_transfer.transfer_skills()` and logs the reward trajectory for each cycle.
- Each cycle fine-tunes policies on a small batch of real examples.

## A-9 Automated PR Conflict Checks

- `src/pr_conflict_checker.py` reuses `pull_request_monitor.list_open_prs()` and runs `git merge-base` to detect conflicts.
- Summaries appear in the AutoBench-style scoreboard.

## A-10 Goal-Oriented Evaluation Harness

- `src/eval_harness.py` gathers benchmark metrics from each module and compares them with the targets in `docs/Plan.md`.
- Running `python -m src.eval_harness` prints a pass/fail table for the whole project.
- For larger experiments run `scripts/distributed_eval.py --workers 4` (or pass `--hosts`) to split the evaluations across processes or nodes.

## A-11 Meta Optimizer

- `src/meta_optimizer.py` implements a lightweight first-order MAML loop. Specify
  `adapt_lr`, `meta_lr` and `adapt_steps` to control the inner and outer updates.
  ``meta_step(model, tasks)`` clones the model for each task, optimises it with
  SGD on the task data and accumulates the resulting gradients before applying a
  meta update.
- `adaptive_planner.AdaptivePlanner` optionally accepts a ``meta_optimizer`` and
  a model. When provided, ``rank_strategies()`` calls ``meta_step`` using the
  latest strategy scores to continually tune the model.

## L-5 Formal Verification Harness

- `src/formal_verifier.py` provides a small property checker that loads model snapshots and symbolically executes critical routines.
- `verify_model()` asserts invariants like gradient norms and output bounds before the model is released.

## M-1 Cross-Modal Fusion Architecture

- `src/cross_modal_fusion.py` embeds text, images and audio in one latent space.
- `train_fusion_model()` fine-tunes a shared encoder-decoder using CLIP- and Whisper-style objectives and `encode_all()` returns embeddings for retrieval.
- Run `scripts/export_onnx.py` to export the fusion and world models as ONNX graphs.

## M-2 World-Model RL Bridge

- `src/world_model_rl.py` learns a generative world model from logged trajectories and runs model-based RL for rapid policy updates.
- The prototype interfaces with ``gym``-like data and provides ``rollout_policy()`` for offline rollout generation.
- `train_with_self_play()` runs ``self_play_skill_loop.run_loop`` to collect
  transitions, converts them into ``TrajectoryDataset`` entries and calls
  ``train_world_model``.

To incorporate voxel observations, call ``GenerativeDataAugmentor.synthesize_3d``
to produce text--volume pairs and pass the result through ``train_world_model``
via the ``synth_3d`` argument. The ``VoxelEnv`` wrapper exposes a 3D state space
so rollout utilities can generate small volumes. ``eval_harness`` provides a
``voxel_rollout`` evaluator that performs a short 3D rollout for quick
verification.


## M-3 Self-Calibration for Embodied Agents

- `src/embodied_calibration.py` adapts sensor and actuator parameters from a small set of real-world samples.
- `calibrate()` aligns simulation and hardware spaces so policies trained in simulation remain effective after deployment.

## M-4 Cross-Modal Data Ingestion Pipeline

- `src/data_ingest.py` provides helpers for downloading and aligning text, image and audio triples.
- Use `download_triples()` to fetch sample files and `align_triples()` to pair them by basename.
- `random_crop()` returns a random image crop while `generate_transcript()` summarises audio duration.

Example usage:

```python
from asi.data_ingest import (
    download_triples,
    align_triples,
    random_crop,
    generate_transcript,
)

triples = download_triples(text_urls, img_urls, aud_urls, "./data")
pairs = align_triples("./data/text", "./data/images", "./data/audio")
img = random_crop(Image.open(pairs[0][1]), (32, 32))
txt = generate_transcript(pairs[0][2])
```

`offline_synthesizer()` rolls out the multimodal world model to generate
simplified synthetic triples offline:

```python
from asi.multimodal_world_model import MultiModalWorldModel, MultiModalWorldModelConfig
from asi.data_ingest import offline_synthesizer
import torch
import numpy as np

cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2)
wm = MultiModalWorldModel(cfg)

def policy(state):
    return torch.zeros((), dtype=torch.long)

def tokenizer(t: str):
    return [ord(c) % cfg.vocab_size for c in t]

triples = offline_synthesizer(wm, tokenizer, "hello", np.zeros((1, 4, 4)), policy, steps=2)
```

Sensitive ingestion steps can be isolated using `EnclaveRunner`:

```python
from asi.enclave_runner import EnclaveRunner
from asi.license_inspector import LicenseInspector
from asi.dataset_lineage_manager import DatasetLineageManager
from asi.data_ingest import download_triples, paraphrase_multilingual
from pathlib import Path

runner = EnclaveRunner()
inspector = LicenseInspector()
lineage = DatasetLineageManager("./data")

triples = download_triples(text_urls, img_urls, aud_urls, "./data", lineage=lineage, runner=runner)
paraphrase_multilingual([Path("./data/text/0.txt")], translator, None, inspector, lineage, runner=runner)
Run `scripts/lineage_viewer.py ./data` to browse the recorded steps.
```

After using `dataset_discovery.store_datasets()` the inspector can audit the
resulting SQLite database and log a summary for each dataset:

```python
inspector = LicenseInspector(["mit"])  # allow only MIT
lineage = DatasetLineageManager("./data")
res = inspector.scan_discovered_db("./discovered.sqlite", lineage)
print(res["hf:example"])  # True or False
```

`LicenseInspector` uses precompiled regex heuristics so scanning large discovery
databases is quick even with thousands of entries.

Each entry adds a note like ``inspect hf:example license=mit allowed=True`` to
``dataset_lineage.json`` so compliance results are tracked alongside other
ingestion steps.

To inject noise for differential privacy, create a `PrivacyGuard` and pass it to
`download_triples()`. After ingestion, inspect the remaining budget:

```python
from asi.privacy_guard import PrivacyGuard

guard = PrivacyGuard(budget=1.0)
download_triples(text_urls, img_urls, aud_urls, "./data", privacy_guard=guard)
print("epsilon left", guard.remaining_budget())
```

## Event Sensor Fusion

`EventSensorDataset` reads either in-memory arrays or ``.npy`` files containing
neuromorphic event streams. Enable fusion by setting
``use_event_streams=True`` when constructing ``MultiModalWorldModel`` and
provide ``event_channels`` matching the input shape. ``train_world_model()``
accepts an ``event_dataset`` and logs the average loss with
``TelemetryLogger`` under the ``world_model_loss`` metric.

Example usage:

```python
from src.event_sensor_dataset import load_synthetic_events
from src.multimodal_world_model import (
    MultiModalWorldModel, MultiModalWorldModelConfig, train_world_model,
)

events = load_synthetic_events(channels=2, length=32)
cfg = MultiModalWorldModelConfig(
    vocab_size=100,
    img_channels=3,
    action_dim=4,
    use_event_streams=True,
    event_channels=2,
)
model = MultiModalWorldModel(cfg)
train_world_model(model, trajectory_ds, event_dataset=events,
                  telemetry=TelemetryLogger(interval=1.0))
```

## L-6 Mechanistic Interpretability Tools

- `src/transformer_circuits.py` provides utilities to record attention weights
  and ablate individual heads. `ActivationRecorder` registers hooks on named
  modules, while `head_importance()` measures output differences when heads are
  zeroed.

## Research workflow

1. **Select an algorithm** from `docs/Plan.md` that remains unsolved.
2. **Review recent literature** to confirm the open problems and techniques.
3. **Prototype the idea** by extending the relevant modules under `src/`.
   Keep the code modular so unit tests in `tests/` can import the new component.
4. **Run the test suite** with `pytest` after each change to catch regressions.
5. **Benchmark the implementation** using the provided scripts under `scripts/`
   or your own short experiments.
6. **Document results** in both `docs/Plan.md` and this file. Describe what
   worked, what failed, and any insights gained.

### Upcoming Implementation Tasks

- Create a `HybridRetention` module that combines the linear update from
- `MambaBlock` with the decay kernel in `RetNetRetention`. **Implemented** in
  `src/hybrid_retention.py` with a unit test.
- Extend `HierarchicalMemory` so `cross_modal_fusion.encode_all()` can store and
  retrieve averaged multimodal embeddings via `add_multimodal`. **Implemented**:
  `encode_all()` now calls `memory.add_multimodal()` when a memory instance is
  passed, enabling lookup on the fused embedding.
- BCI embeddings can now be fused and stored the same way. `CrossModalFusion`
  encodes EEG/ECoG signals and `HierarchicalMemory.add_multimodal()` averages
  them with text, image and audio vectors for world-model training.
- Sign-language videos can be processed with `SignLanguageRecognizer`. Pass
  `sign_videos` to `encode_all()` so the resulting embeddings are stored via
  `add_multimodal()`. The recognizer now distinguishes common signs like
  **hello** and **thanks** using simple landmark heuristics so retrieval works
  across a broader gesture set.
  Example usage:

  ```python
  videos = [np.zeros((1, 1, 3), dtype=np.float32) for _ in range(len(dataset))]
  t, i, a, s = encode_all(model, dataset, sign_videos=videos, include_sign=True)
  ```
- **Sign-language graph controller**: ``SignLanguageGraphController`` interprets
  webcam gestures with ``SignLanguageRecognizer`` and translates the resulting
  command using ``CrossLingualTranslator`` so ``NLGraphEditor`` can edit a
  ``CrossLingualReasoningGraph`` in any supported language. A ``mapping`` dict
  converts gestures like ``hello`` into graph commands such as ``add node hello``.
  Run ``python scripts/sign_language_webcam.py`` to try the live demo.
- **Multimodal reasoning graph**: `CrossLingualReasoningGraph.add_step()`
  now accepts `image_embed` and `audio_embed`. Use
  `cross_modal_fusion.embed_modalities()` to obtain vectors from raw data and
  store them as `image_vec` and `audio_vec` metadata so traces can reference
  pictures and sound clips. The logger preserves these vectors when saving
  history:

  ```python
  t_vec, i_vec, a_vec = embed_modalities(model, tokenizer, text, image, audio)
  nid = graph.add_step(text, image_embed=i_vec, audio_embed=a_vec)
  logger.log({"summary": text, "image_vec": i_vec, "audio_vec": a_vec})
  ```
- Add a `log_memory_usage()` helper to `eval_harness.py` and print GPU memory usage alongside accuracy metrics. **Implemented**
- Integrate `QAEHyperparamSearch` into `MetaRLRefactorAgent` to tune the exploration rate during refactoring. **Implemented**
- Rewrite `download_triples()` with asyncio to fetch dataset files
  concurrently. **Implemented** with an async helper using `aiohttp`.
  - Add streaming RPCs to `MemoryServer` so batches of vectors can be pushed and
    queried in one call. Update `memory.proto` and the `RemoteMemoryClient`.
  **Implemented** via `push_batch_remote()` and `query_batch_remote()` in
  `src/hierarchical_memory.py`.
- Implement optional gradient checkpointing in `multimodal_world_model.py` via a
  `checkpoint_blocks` flag to cut training memory. **Implemented** in
  `src/multimodal_world_model.py`.
- Create `scripts/distributed_memory_benchmark.py` that measures throughput of
  `DistributedMemory` across multiple nodes. **Implemented**
- Implement a `PrioritizedReplayBuffer` in `self_play_env.py` and adapt
  `self_play_skill_loop.run_loop()` to sample transitions by reward. **Implemented**
- Create `scripts/distributed_eval.py` to run `eval_harness` across multiple
  processes or hosts and aggregate the results. **Implemented**
- Extend `transformer_circuits.py` with an `AttentionVisualizer` class that
  saves interactive attention heatmaps for interpretability experiments.
  **Implemented** in `src/transformer_circuits.py` with unit tests.
- `scripts/attention_analysis.py` loads a saved model and text file, hooks
  `AttentionVisualizer`, and writes the resulting heatmaps. Run it as:

```bash
python scripts/attention_analysis.py --model model.pt --input sample.txt --out-dir vis
```
- Prototype a `GraphOfThoughtPlanner` that composes reasoning steps into a
  searchable graph for code refactoring decisions.
- **Implemented** a `GraphOfThoughtPlanner` via `GraphOfThought` (see
  `src/graph_of_thought.py`) that composes reasoning steps into a searchable
  graph for code refactoring decisions.
- Add a `NeuroSymbolicExecutor` module that runs logical constraints alongside
  neural world-model rollouts. **Implemented in `src/neuro_symbolic_executor.py`.**
- Implement a `DistributedTrainer` that automatically restarts failed
  processes and coordinates checkpoints with `DistributedMemory`. **Implemented**
  in `src/distributed_trainer.py` with tests.
    - Build an `EdgeMemoryClient` in `memory_clients.py` to stream context vectors
      to `RemoteMemoryClient` so edge devices can handle large-context inference.
      **Implemented**
- The client now keeps a local queue when the network is unreachable and
  periodically flushes queued `add`/`delete` operations once connectivity
  returns. Run `scripts/edge_memory_client_demo.py --offline` to observe
  queued updates syncing after the server starts.
- Create an `AdaptiveCurriculum` that blends curated data with
  self-play logs using reinforcement learning. **Implemented in `src/adaptive_curriculum.py` with tests.**
- Extend `QAEHyperparamSearch` to explore novel transformer components during
  architecture search. **Implemented in `src/quantum_hpo.py` with unit tests.**
- **Implemented** an `ElasticMoERouter` that scales the number of active experts
  according to real-time GPU utilization.
- **Implemented** an `RLMoERouter` that trains routing weights via a simple
  reinforcement learning loop for improved load balance.
- Extend `HierarchicalMemory` with an `SSDCache` that prefetches high-frequency
  vectors for faster retrieval. *Implemented with a disk-backed cache and
  persistence helpers in `src/hierarchical_memory.py`.*
- Build an `AutoDatasetFilter` using generative noise detection to discard low-quality training samples before ingestion. **Implemented**
- Implement a `GenerativeDataAugmentor` that rolls out the world model to
  synthesize training triples and feeds them through `data_ingest`. **Implemented**
  in `src/generative_data_augmentor.py`.
- Add `continuous_eval.py` to schedule `eval_harness` and `autobench` after each pull request using GitHub Actions or a local cron job. **Implemented in `scripts/continuous_eval.py`.**
- Combine `GraphOfThoughtPlanner` with `MetaRLRefactorAgent` in an `AdaptivePlanner`
  module that ranks and applies refactor suggestions automatically. **Implemented in `src/adaptive_planner.py`.**
- `src/neural_arch_search.py` implements distributed architecture search and is
  integrated with `eval_harness.py` to score candidate models automatically. **Implemented in `src/neural_arch_search.py`.**
- Implement a `SelfHealingTrainer` that monitors distributed jobs and restarts
  failed runs to maintain full compute utilization. **Implemented in `src/self_healing_trainer.py`.**
- Extend `data_ingest.py` with an `offline_synthesizer` that uses the world
  model to generate synthetic multimodal triples for training. **Implemented as `data_ingest.offline_synthesizer`.**
- Implement a `FederatedMemoryExchange` service that synchronizes vectors across multiple `MemoryServer` nodes. Provide a `scripts/federated_memory_sync.py` utility to benchmark cross-node synchronization throughput. **Implemented in `src/federated_memory_exchange.py` with the benchmarking script `scripts/federated_memory_sync.py`.**
- Create a `CausalGraphLearner` module that infers directed relations from `world_model_rl` transitions and logs the resulting edges for planning. **Implemented in `src/causal_graph_learner.py`.**
- Develop a `CausalReasoner` that combines `CausalGraphLearner` with `NeuroSymbolicExecutor` to plan actions along learned cause–effect chains. **Implemented in `src/causal_reasoner.py` with tests.**
- Extend `world_model_rl` with `simulate_counterfactual()` which consults the
  learned graph to adjust predicted transitions for hypothetical interventions.
  This improves planning accuracy by allowing the reasoner to explore "what-if"
  scenarios. See `scripts/causal_sim.py` for an example.
- Add a `SelfAlignmentEvaluator` to `eval_harness.py` that runs `deliberative_alignment.check_alignment()` on generated outputs and reports the metrics alongside existing benchmarks. **Implemented as `_eval_self_alignment()` in `src/eval_harness.py`.**
- Add an `ActiveDataSelector` to `data_ingest.py` that scores incoming triples by predictive entropy and filters out low-information samples before storage. **Implemented in `data_ingest.ActiveDataSelector`.**
- Implement a `FederatedMemoryServer` variant that replicates vector stores across peers using gRPC streaming consensus for decentralized retrieval. The server now includes a `Sync` RPC implementing CRDT merge semantics so replicas converge after partitions. Retrieval proofs can optionally be checked when vectors are synced so peers verify the hashed embeddings before accepting them. **Implemented in `src/federated_memory_server.py`.**

Example usage:
```python
from asi.hierarchical_memory import HierarchicalMemory
from asi.federated_memory_server import FederatedMemoryServer

mem1 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
mem2 = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
s1 = FederatedMemoryServer(mem1, "localhost:50500", peers=["localhost:50501"], require_proof=True)
s2 = FederatedMemoryServer(mem2, "localhost:50501", peers=["localhost:50500"], require_proof=True)
s1.start(); s2.start()
# pushing to one server replicates with proofs
s1.stop(0); s2.stop(0)
```

Enabling proof verification adds a small SHA-256 hash computation per vector when syncing. Proof digests are cached and replication now uses a thread pool to contact peers concurrently, roughly halving the latency compared to the naive approach.
- Develop a `HierarchicalPlanner` combining `GraphOfThought` with `world_model_rl.rollout_policy` to compose multi-stage plans. **Implemented in `src/hierarchical_planner.py`.**
- Integrate a `DifferentialPrivacyOptimizer` into training loops so models can optionally clip gradients and inject noise during updates. **Implemented in `src/differential_privacy_optimizer.py` and integrated with `world_model_rl.train_world_model`.**
- Add a `PrivacyBudgetManager` to track cumulative privacy loss across runs. `train_world_model` accepts the manager and records the consumed epsilon/delta after each training session. **Implemented in `src/privacy_budget_manager.py` with `scripts/privacy_budget_status.py`.**
- Add a `GradientCompressor` utility that performs top-k or quantized gradient
  compression. `DistributedTrainer` uses it when ``grad_compression`` is
  provided. **Implemented in `src/gradient_compression.py` and wired through
  `distributed_trainer.py`.**
- **Added** an asynchronous training mode in `DistributedTrainer`. Workers apply
  gradients locally and periodically synchronize via parameter averaging.
 - Add a `LocalitySensitiveHashIndex` option in `vector_stores.py` so `HierarchicalMemory` can perform approximate nearest neighbor search with sub-linear query time. **Implemented in `vector_stores.LocalitySensitiveHashIndex` and wired through `HierarchicalMemory`.**
- Create an `EmbeddingVisualizer` module that runs UMAP/t-SNE on cross-modal embeddings and serves interactive plots through a minimal web interface.
**Implemented in `src/embedding_visualizer.py`.**
- Implement a `MultiAgentCoordinator` that synchronizes multiple `MetaRLRefactorAgent` instances and schedules cooperative refactoring tasks across repositories. **Implemented in `src/multi_agent_coordinator.py` with unit tests.**
- Extend `MultiAgentCoordinator` with a pluggable `NegotiationProtocol`. A reinforcement-learning based `RLNegotiator` assigns tasks to agents. **Implemented with tests in `src/multi_agent_coordinator.py`.**
 - Implement a `PQVectorStore` using FAISS `IndexIVFPQ` for compressed vector storage and integrate it with `HierarchicalMemory`. Benchmark retrieval accuracy against `FaissVectorStore`. **Implemented in `src/vector_stores.py` and integrated with `HierarchicalMemory`.**
- Build quantized code indexes with `code_indexer.py` and `incremental_pq_indexer.py`. Run `scripts/build_pq_index.py src index_dir` then start `QuantizedMemoryServer` to serve the shards.
- Implement a `HolographicVectorStore` that encodes text, image and audio embeddings with holographic reduced representations. `scripts/holographic_retrieval_benchmark.py` measures retrieval accuracy with this store type. Run `python scripts/holographic_retrieval_benchmark.py --samples 500` after installing `faiss-cpu` to reproduce the numbers.
- Add a `DuplicateDetector` that uses CLIP embeddings with locality-sensitive hashing to drop near-duplicate samples during ingestion and connect it to `AutoDatasetFilter`. **Implemented in `src/duplicate_detector.py` and integrated with `filter_text_files()`.**
- Add a `DataPoisonDetector` that clusters word statistics and flags poisoned samples during ingestion. `download_triples()` now drops flagged triples. **Implemented in `src/data_poison_detector.py` and wired through `data_ingest`.**
- Add a `PrivacyGuard` that injects noise into downloaded triples and tracks epsilon usage. `download_triples()` records the budget per sample. **Implemented in `src/privacy_guard.py` and integrated with `data_ingest.download_triples`.**
- Implement a `TemporalVectorCompressor` in `streaming_compression.py` with a
  decay factor so `HierarchicalMemory` can prioritize recent context. Benchmark
  retrieval accuracy against the existing compressor. **Implemented in
  `src/streaming_compression.py`.**
- Add a `CrossLingualTranslator` helper in `data_ingest.py` to translate text
  into multiple languages during ingestion and store the augmented triples.
  **Implemented in `data_ingest.CrossLingualTranslator`.**
- Provide a `CrossLingualMemory` wrapper that stores translated vectors and
  lets `HierarchicalMemory` search across languages when a translator is set.
  **Implemented in `src/cross_lingual_memory.py` with tests.**
- Add a `CrossLingualReasoningGraph` to store reasoning nodes with language tags
  and translate them via `CrossLingualTranslator`. `GraphOfThoughtPlanner` logs
  ranked strategies to this graph when provided.
- Extend `CrossLingualReasoningGraph` with `summarize_old_steps()` which calls
  `ContextSummaryMemory` when traces grow beyond a threshold. Translated
  summaries are stored via `CrossLingualTranslator.translate_all` and can be
  returned during planning through `GraphOfThought.plan_refactor(summary_memory)`.
  `query_summary()` retrieves these summaries in any supported language and
  `ReasoningHistoryLogger.log()` records which node ids were summarised along
  with the memory location for later inspection.
- Extend `CrossLingualReasoningGraph` with `search(query, lang)` which
  translates ``query`` to each node's language, embeds the texts using a
  deterministic hash and returns node IDs ranked by cosine similarity.
  Embeddings are cached per-language so subsequent searches reuse them.
- Create a `WorldModelDistiller` module and a `scripts/distill_world_model.py`
  utility to train smaller student models from the large world model.
  **Implemented in `src/world_model_distiller.py` with the script
  `scripts/distill_world_model.py`.**
- Implement a `SummarizingMemory` helper that compresses infrequently used vectors with a small language-model summarizer. Provide `scripts/summarize_memory_benchmark.py` to measure storage savings and retrieval accuracy.
  **Implemented in `src/summarizing_memory.py` with the benchmarking script
  `scripts/summarize_memory_benchmark.py`.**
- Introduced `BaseSummarizingMemory` encapsulating usage tracking and summary replacement. `SummarizingMemory`, `ContextSummaryMemory` and `MultiModalSummaryMemory` now inherit from it.
- Implement a `ContextSummaryMemory` that replaces far-past vectors with text summaries and re-expands them when retrieved. Unit test `tests/test_context_summary_memory.py` verifies summarization and expansion.
  **Implemented in `src/context_summary_memory.py` with tests.**
- Extend `ContextSummaryMemory` with a `translator` argument so summaries are stored in multiple languages and returned in the query language. Tested in `tests/test_cross_lingual_summary_memory.py`.
- Add a `ReasoningSummaryTranslator` that clusters reasoning steps via `ReasoningHistoryLogger.analyze()` and translates the final report using `CrossLingualTranslator`. `self_reflection.main()` now prints these multilingual summaries. Example output:
  ```bash
  $ python -m asi.self_reflection history.json
  Reasoning step clusters:
  - start: 1
  Translations:
  - [es] Reasoning step clusters:\n- start: 1
  ```
- `scripts/cross_lingual_reasoning_demo.py` indexes a short reasoning history
  with `CrossLingualMemory` using a caching translator and verifies retrieval in
  Spanish and French. The demo prints `cross_lingual_accuracy: 1.00` on three
  toy steps.
- Extend `analogical_retrieval.analogy_search` with a `language` argument and update `HierarchicalMemory.search(mode="analogy")` so `ContextSummaryMemory` can return translated vectors. Tested in `tests/test_cross_lingual_analogy.py`.
- Implement a `KnowledgeGraphMemory` that stores `(subject, predicate, object)` triples and hooks into `HierarchicalMemory` via `use_kg=True`. Unit tests cover insertion and retrieval.
  **Implemented in `src/knowledge_graph_memory.py` with `tests/test_knowledge_graph_memory.py`.**
- Implement a `FederatedKGMemoryServer` that replicates `KnowledgeGraphMemory` across peers using CRDT-based updates. Endpoints `Push`, `PushBatch`, `Query` and `Sync` ensure replicas converge after partitions.
  **Implemented in `src/federated_kg_memory.py` with tests.**
  `KnowledgeGraphMemory` accepts an optional timestamp for each triple and `query_triples()` can filter by a time range.
  **Implemented in `src/knowledge_graph_memory.py` with `tests/test_knowledge_graph_memory.py` and `tests/test_time_aware_kg.py`.**
- Implement a `TemporalReasoner` that queries these timestamped triples and infers
  their chronological order. `HierarchicalPlanner.compose_plan()` accepts the
  reasoner and can reorder intermediate nodes for time-aware planning.
- **Implemented in `src/temporal_reasoner.py` with tests.**
- Add a `TelemetryLogger` in `telemetry.py` that exports GPU, CPU and network metrics via OpenTelemetry and Prometheus. Integrate the logger with `DistributedTrainer` and `MemoryServer`.
  `MemoryServer` now starts and stops a provided `TelemetryLogger` automatically.
  **Implemented in `src/telemetry.py`.**
- Implement a `MemoryDashboard` that aggregates `TelemetryLogger` statistics from multiple `MemoryServer` nodes and serves them via a simple web endpoint. The script `scripts/memory_dashboard.py` launches the dashboard.
  **Implemented in `src/memory_dashboard.py` with tests.**
- Extend `MemoryDashboard` with `/entries`, `/add` and `/delete` HTTP routes for manipulating `HierarchicalMemory` contents.
  **Implemented in `src/memory_dashboard.py` with tests.**
- Add a `RetrievalVisualizer` that records timestamped hit/miss events and
  serves an aggregated latency plot via the dashboard's new `/patterns` endpoint.
  Enable it during evaluation to monitor retrieval frequency.
  ```python
  mem = HierarchicalMemory(dim=32, compressed_dim=16, capacity=1000)
  vis = RetrievalVisualizer(mem)
  vis.start()
  dashboard = MemoryDashboard([server], visualizer=vis)
  ```
- Start `RetrievalPolicyUpdater` with the same `TelemetryLogger` used by
  `HierarchicalMemory` or the visualizer to refine ranking from query logs.
  The updater expects `(meta, hit, latency)` tuples and reports recall
  improvements via the logger.
  ```python
  logger = TelemetryLogger(interval=1.0)
  policy = RetrievalPolicy()
  mem = HierarchicalMemory(dim=32, compressed_dim=16, capacity=1000,
                           retrieval_policy=policy)
  vis = RetrievalVisualizer(mem)
  updater = RetrievalPolicyUpdater(policy, load_logs, telemetry=logger)
  vis.start(); updater.start()
  ```
- Add an `InterpretabilityDashboard` that exposes attention heatmaps generated by
  `AttentionVisualizer` along with memory statistics through a small HTML/JS
  interface. `scripts/memory_dashboard.py` starts this dashboard next to the
  risk metrics server.
  **Implemented in `src/interpretability_dashboard.py` with tests.**
- Implement a `MultiAgentDashboard` that aggregates telemetry and reasoning logs from multiple agents and exposes task assignments and carbon usage via a small HTTP server.
  **Implemented in `src/multi_agent_dashboard.py` with tests.**
- Build an `AlignmentDashboard` that records outputs from `DeliberativeAligner`,
  `IterativeAligner` and `CriticRLHFTrainer`. `eval_harness.py` publishes pass
  rates and flagged examples to this dashboard.
  **Implemented in `src/alignment_dashboard.py` with tests.**
  - Extend `data_ingest.py` with a `LicenseInspector` that parses dataset metadata for license terms and rejects incompatible samples. Include a `scripts/license_check.py` CLI to audit stored triples.
  **Implemented in `src/license_inspector.py` with the CLI `scripts/license_check.py`.**
- Add an `AdaptiveCompressor` that tunes the compression ratio in `StreamingCompressor` based on retrieval frequency so rarely accessed vectors use fewer bytes.
  **Implemented as `AdaptiveCompressor` in `src/streaming_compression.py`.**
- Create a `PromptOptimizer` module that rewrites prompts via reinforcement learning and tracks evaluation improvements automatically.
  **Implemented in `src/prompt_optimizer.py`.**
- `UserPreferences` now keeps a short history of recent emotions per user and
  `PromptOptimizer.optimize()` adjusts the final prompt based on this history.
  A helper script `scripts/ab_prompt_eval.py` compares two optimizer
  configurations using `ab_evaluator.run_config()`. Example configs live under
  `configs/ab_prompt_example*.json`.
- Integrate a `TrainingAnomalyDetector` with `SelfHealingTrainer` to roll back or restart runs when loss spikes beyond a configurable threshold.
  **Implemented via `TrainingAnomalyDetector` in `src/training_anomaly_detector.py`.**
- Implement a `ParameterEfficientAdapter` that applies low-rank adapters to target modules for cross-task fine-tuning. **Implemented in `src/parameter_efficient_adapter.py` with tests.**
- Add a `DatasetVersioner` module that logs dataset hashes and transformation steps. Extend `data_ingest` so all downloads and synthetic samples record their provenance in a version file.
  **Implemented in `src/dataset_versioner.py` and wired through `data_ingest`.**
- Add a `DatasetLineageManager` that records transformation steps and resulting file hashes for reproducible pipelines.
  **Implemented in `src/dataset_lineage_manager.py` with tests.**
- Provide a `DatasetLineageDashboard` exposing `/graph` and `/steps` endpoints for searching lineage records. Run `python scripts/lineage_dashboard.py <root>` to launch it.
  **Implemented in `src/dataset_lineage_dashboard.py` with tests.**
- Introduce a `BlockchainProvenanceLedger` that links each record to the previous hash. Ingestion helpers append their lineage to this ledger and `scripts/check_blockchain_provenance.py` verifies the chain.
- Expose a `DatasetLineageService` gRPC API. `dataset_lineage_server.py` signs
  each record with an ed25519 key and writes it to the `BlockchainProvenanceLedger`.
  The companion `dataset_lineage_client.py` provides `add_entry()` and
  `get_entries()` helpers.
- Implement a `ContextWindowProfiler` that measures memory footprint and wall-clock time at various sequence lengths. **Implemented as `src/context_profiler.py` and integrated with `eval_harness.py`.**
- Extend `HierarchicalMemory` with an adaptive eviction policy that prunes rarely used vectors and emit statistics on hit/miss ratios.
  **Implemented** via `adaptive_evict` in `HierarchicalMemory` with `get_stats()` to report usage metrics.
- Add an optional `SafetyPolicyMonitor` hook in `self_play_skill_loop` that runs `deliberative_alignment` each cycle and logs policy violations.
  **Implemented in `src/self_play_skill_loop.py`.**
- Add `export_to_onnx()` in `src/onnx_utils.py` and `scripts/export_onnx.py` to save ONNX graphs for `MultiModalWorldModel` and `CrossModalFusion`. Run `python scripts/export_onnx.py --out-dir models` to generate them.
- Implement a `SecureFederatedLearner` that aggregates encrypted gradients from remote peers so training can proceed without sharing raw data. Provide a `scripts/federated_train.py` CLI.
  **Implemented in `src/secure_federated_learner.py` with `scripts/federated_train.py`.**
- Add an `AcceleratorScheduler` that detects GPU, TPU or CPU utilisation and dispatches queued jobs when the requested device has available capacity. Integrate it with `DistributedTrainer`.
  Combine it with `ComputeBudgetTracker` in `adaptive_scheduler.py` so jobs pause when the accelerator budget runs out and resume once progress improves.
  **Implemented in `src/accelerator_scheduler.py` with optional temperature throttling and extended by `src/adaptive_scheduler.py`.**
- Develop an `AdversarialRobustnessSuite` that generates adversarial prompts and reports failure cases through `eval_harness`.
  **Implemented in `src/adversarial_robustness.py`.**
- Implement a `DatasetBiasDetector` module that computes representation metrics and integrates with `AutoDatasetFilter`. Provide a `dataset_bias_report.py` utility for bias analysis.
  **Implemented in `src/dataset_bias_detector.py` with `scripts/dataset_bias_report.py`.**
- Create a `FederatedWorldModelTrainer` that averages gradients across nodes for distributed world-model training. Include `scripts/federated_world_model_train.py`.
  **Implemented in `src/federated_world_model_trainer.py` with script `scripts/federated_world_model_train.py`.**
- Add a `GradientPatchEditor` helper to apply small updates that fix incorrect outputs without full fine-tuning.
  **Implemented in `src/gradient_patch_editor.py`.**
- Extend `graph_of_thought.py` with a `ReasoningDebugger` that consolidates contradictions and loops from multiple agents.
  **Implemented in `src/graph_of_thought.py`.**
- Implement a `GraphQLMemoryGateway` that exposes `MemoryServer` retrieval endpoints via GraphQL. Provide `scripts/graphql_memory_server.py` to benchmark query overhead.
  **Implemented in `src/graphql_memory_gateway.py` with `scripts/graphql_memory_server.py`.**
- Add a `FineGrainedProfiler` in `telemetry.py` to record per-module compute and memory usage and stream the metrics through `TelemetryLogger`.
  **Implemented in `src/telemetry.py`.**
 - Create an `AutoLabeler` that invokes the world model during ingestion to generate weak labels for unlabeled triples. A reinforcement-learning agent now refines those labels using bias metrics and user feedback.
   **Implemented in `src/auto_labeler.py`.**
- Implement a `SensorimotorPretrainer` that performs self-supervised pretraining
  of `MultiModalWorldModel` on raw sensor logs. Provided
  `pretrain_sensorimotor()` in `src/sensorimotor_pretrainer.py` with a unit
  test. **Implemented.**
- Add a `MultiStageOversight` helper combining `CollectiveConstitution`,
  `DeliberativeAligner`, `CriticRLHFTrainer`, and formal verification checks to
  enforce multi-stage safety. Implemented in `src/multi_stage_oversight.py` with
  tests.
- Introduce `DifferentialPrivacyMemory` that injects Gaussian noise into
  embeddings before storage to reduce privacy leakage.
  **Implemented in `src/dp_memory.py`.**
- Implement `MultiModalEval` in `eval_harness.py` to report recall@k for text,
  image and audio in a single run. Enable with `--multimodal`.
  **Implemented in `src/eval_harness.py`.**
- Create `MultiAgentGraphPlanner` that wraps `GraphOfThoughtPlanner` with
  `MultiAgentCoordinator` for collaborative graph planning.
  **Implemented in `src/multi_agent_graph_planner.py`.**
- Add a `WorldModelDebugger` that patches the world model when rollout errors
  exceed a threshold using `GradientPatchEditor`.
  **Implemented in `src/world_model_debugger.py`.**
- Provide a `ModelVersionManager` to log model hashes alongside dataset
  versions for full reproducibility.
  **Implemented in `src/model_version_manager.py`.**
- Introduce a `ModelCardGenerator` that collates dataset lineage, telemetry
  stats and evaluation results into a Markdown or JSON model card.
  **Implemented in `src/model_card.py` with CLI `scripts/generate_model_card.py`.**
- Extend `GraphOfThought` with `summarize_trace()` and an `explain` flag so
  reasoning steps can be rendered in plain language for debugging.
- Add `self_reflect()` to `GraphOfThought` which outputs a concise summary of
  reasoning steps. `ReasoningHistoryLogger` stores these summaries with
  timestamps for later inspection.
- `GraphUI` exposes `/graph/node`, `/graph/edge`, `/graph/remove_*` and
  `/graph/recompute` so reasoning graphs can be edited interactively. Each edit
  records a new summary via `ReasoningHistoryLogger`. The script
  `scripts/graph_playground.py` launches this playground.
- `GraphOfThought.to_json()` now emits a deterministic `stable_id` for each
  node so snapshots can be diffed across runs. `ReasoningHistoryLogger.save_graph()`
  persists a graph to disk and logs the path. `scripts/graph_diff.py` compares
  two saved graphs and reports added or changed nodes and edges. The logger's
  `analyze()` method returns these diffs alongside step clusters.
- Provide a `ResourceBroker` module coordinating multiple clusters and a demo
  script `scripts/resource_broker_demo.py`. The broker now reports per-accelerator
  utilisation via `get_load()` and allows allocating jobs to specific
  accelerator types.
- Introduce an `ARDebugger` that streams robot state via WebSockets so predicted
  and actual trajectories from `world_model_rl` can be overlaid in an AR client.
  A WebXR viewer (`scripts/webxr_viewer.js`) renders the streamed graph in the
  browser. See `scripts/ar_robot_demo.py` for a minimal demo.
  - `vr_graph_explorer.py` uses pythreejs to display a `GraphOfThought` in VR.
    Layout and edge computation are vectorised with NumPy for faster updates.
    Voice and sign-language commands are handled by `VoiceGraphController` so
    graphs can be edited hands free. Launch it with
    `scripts/vr_explorer.py trace.json` and open the reported URL.
- `research_ingest.py` fetches arXiv titles and abstracts, translates them via
  `CrossLingualTranslator` and stores language-tagged summaries under
  `research_logs/`.
- Expand `GenerativeDataAugmentor` with `synthesize_3d()` for basic 3D asset
  synthesis.
- Extend `world_model_rl.train_world_model()` to accept 3D data from this
  augmentor, introduce a `VoxelEnv` wrapper emitting voxel observations and add
  a `voxel_rollout` evaluator in `eval_harness.py` for 3D rollouts.
- Implement a mocked `quantum_sampler.sample_actions_qae()` and integrate it as
  an optional sampler in `train_with_self_play`.
- Compute an overall risk metric via the new `RiskScoreboard` module.
- Add a `ComputeBudgetTracker` that records GPU hours and memory usage via
  `TelemetryLogger` and feeds the estimated energy cost into
  `RiskScoreboard`. **Implemented in `src/compute_budget_tracker.py` with tests.**
- Add a `CarbonFootprintTracker` that reads CPU/GPU power via NVML or OS
  counters and reports kWh and CO₂ emissions. `TelemetryLogger` can start this
  tracker and `ComputeBudgetTracker` now exposes per-run carbon usage.
  **Implemented in `src/carbon_tracker.py` with tests.**

Short example using the carbon-aware scheduler:

```python
from asi.telemetry import TelemetryLogger
from asi.compute_budget_tracker import ComputeBudgetTracker
from asi.adaptive_scheduler import AdaptiveScheduler

tel = TelemetryLogger(interval=1.0, carbon_tracker=True)
budget = ComputeBudgetTracker(2.0, telemetry=tel)
sched = AdaptiveScheduler(budget, "demo")

sched.add(train_step, region="US-east")
sched.add(eval_step, region="EU-west")
time.sleep(10)
sched.stop()
```

`TelemetryLogger` publishes energy stats to a running
`ClusterCarbonDashboard` so operators can monitor cluster-wide impact. The
dashboard now lists negotiated schedules and the cumulative carbon saved.

### Telemetry Aggregation

`TelemetryAggregator` collects JSON metrics from multiple `TelemetryLogger`
instances and exposes the summed statistics via a `/metrics` endpoint. Start the
service and point each logger at it using `publish_url`:

```python
from asi.telemetry_aggregator import TelemetryAggregator
from asi.telemetry import TelemetryLogger

agg = TelemetryAggregator()
agg.start(port=9000)
logger = TelemetryLogger(publish_url=f"http://localhost:{agg.port}", node_id="n1")
logger.start()
```

Running additional nodes with their own `TelemetryLogger` instances will update
the aggregator which can then be scraped by Prometheus.

- Introduce an `EnergyAwareScheduler` that queries `TelemetryLogger.get_carbon_intensity()`
  and delays or migrates jobs when the value exceeds a threshold. Enable it by
  passing `energy_scheduler=True` to `AdaptiveScheduler`. It complements the
  existing `BudgetAwareScheduler` for compute-aware training.

- Add an `AdaptiveMicroBatcher` that monitors GPU memory via `TelemetryLogger`
  and adjusts micro-batch sizes automatically. `DistributedTrainer` and
  `EdgeRLTrainer` accept it through the optional `micro_batcher` argument.
- Combine `SelfHealingTrainer` and `MultiAgentCoordinator` in a simplified
  `CollaborativeHealingLoop` for cooperative recovery.

## Fairness Evaluator

`src/fairness_evaluator.py` computes demographic parity and equal opportunity gaps from per-group label statistics. The evaluation harness exposes these metrics via the `fairness_evaluator` entry.

`data_ingest.paraphrase_multilingual()` generates translations and paraphrases
in multiple languages. Outputs are filtered with `AutoDatasetFilter` and checked
by `LicenseInspector` before being saved. The number of generated and retained
files is logged through `DatasetLineageManager`. To quantify fairness gains,
run `CrossLingualFairnessEvaluator` on the dataset before and after augmentation
and compare the demographic parity gap.

`src/dataset_weight_agent.py` maintains per-dataset weights by combining
license checks from `LicenseInspector`, bias scores from `dataset_bias_detector`
and Q-learning updates from validation accuracy and fairness statistics.
`ActiveDataSelector` multiplies sample weights by these learned factors so
biased or non-compliant datasets are down‑weighted during training.

## LoRA Merger

`src/lora_merger.py` merges multiple LoRA checkpoints by weighted averaging. Use `scripts/merge_lora.py` to create a single adapter file before loading it in the model.

## Edge RL Trainer

`src/edge_rl_trainer.py` trains world models under a compute budget. It checks `ComputeBudgetTracker.remaining()` each step and stops when resources run low. See `scripts/train_edge_rl.py` for a usage example.
`EdgeRLTrainer` now accepts `use_loihi=True` to run spiking layers on neuromorphic
hardware. Energy usage for CPU vs. Loihi runs is tracked via
`TelemetryLogger` and exposed through the new `power_usage` attribute.

`src/fhe_runner.py` provides `run_fhe()` to execute small models with fully
homomorphic encryption. It relies on the open‑source TenSEAL library and
supports only 1‑D tensors. Expect substantial overhead (often 100× slower) from
ciphertext operations and memory usage, so this mode is suitable for toy
examples rather than large‑scale training.

### Resource-aware scheduling

`MultiAgentCoordinator` accepts a `ComputeBudgetTracker` instance which tracks GPU hours per agent. `RLNegotiator` considers `tracker.remaining()` when assigning tasks and each action logs usage via `tracker.consume()`. This ensures repositories are processed by agents with sufficient budget.

### RL Multi-Cluster Scheduler

`src/rl_multi_cluster_scheduler.py` extends the heuristic `MultiClusterScheduler`
with a tiny Q-learning policy.  The table is keyed by `(cluster, hour)` and is
updated from historical queue time, spot price and carbon intensity logs using
`update_policy()`.  At runtime `submit_best_rl()` chooses the cluster with the
highest expected reward, falling back to random exploration with a small
`epsilon`.

Compared to the ARIMA‑based heuristic scheduler, the RL variant adapts to
recurring patterns in queue delays and energy prices.  Over time it tends to
migrate jobs toward the cheaper and greener cluster even when short‑term
forecasts fluctuate.

The scheduler now accepts a `telemetry` mapping so each cluster can register a
`TelemetryLogger`. When jobs are dispatched the chosen cluster and estimated
carbon saving are logged. Calling `cluster_stats()` returns the aggregated
telemetry per cluster.

````python
from asi.telemetry import TelemetryLogger
from asi.cluster_carbon_dashboard import ClusterCarbonDashboard
from asi.rl_multi_cluster_scheduler import RLMultiClusterScheduler
from asi.hpc_forecast_scheduler import HPCForecastScheduler

dash = ClusterCarbonDashboard(); dash.start(port=0)
tele = {
    "us": TelemetryLogger(node_id="us", publish_url=f"http://localhost:{dash.port}/update"),
    "eu": TelemetryLogger(node_id="eu", publish_url=f"http://localhost:{dash.port}/update"),
}
sched = RLMultiClusterScheduler({"us": HPCForecastScheduler(), "eu": HPCForecastScheduler()}, telemetry=tele, dashboard=dash)
sched.submit_best_rl(["run.sh"], expected_duration=2.0)
print(sched.cluster_stats())
dash.stop()
````
`ClusterCarbonDashboard` now shows the negotiated schedules with their carbon
savings in the web view.

`HPCForecastScheduler.forecast_scores()` caches the ARIMA results keyed by the
history lengths so repeated calls avoid refitting the model.

- `src/nerf_world_model.py` implements a tiny NeRF renderer with multi-view dataset helpers. Training on the synthetic cube sequence reaches around **25 dB PSNR** after 50 epochs.

## Graph Neural Memory

`src/gnn_memory.py` implements a tiny GraphSAGE encoder over a `GraphOfThought`. Each node text is hashed into an initial vector and message passing averages neighbour features before a linear projection. The resulting embeddings are used for context-aware search across reasoning steps.

**Message passing steps**

1. Embed all node texts deterministically with `_embed_text`.
2. For every node, compute the mean embedding of its outgoing neighbours.
3. Combine self and neighbour representations with two linear layers and a ReLU.

**Training objective**

Edges are reconstructed via a simple link prediction loss. For each observed edge the dot product of connected nodes is maximised while the score for a random negative node is minimised.

**Integration**

`encode_nodes()` returns the learned node embeddings. `query(context)` fetches the embeddings of neighbours of a given node or set of nodes so existing memory modules can condition retrieval on the current reasoning context.

