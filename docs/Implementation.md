# Implementation Notes for S-1 and S-2

This repository includes starter modules for the first two algorithms listed in `docs/Plan.md`.

## S-1 Sparse Mixture-of-Experts Routing

- `src/moe_router.py` provides two routers:
  - `HashRouter` uses hash-based gating to activate at most two experts per token.
  - `SwitchRouter` employs a learned linear gate and selects the top-k experts.
  Both expose `load_balance_std` and `expert_utilization` to inspect token distribution.
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

Run it from the project root. By default it uses `HashRouter`; pass `--router switch` to benchmark the learned router:

```bash
python scripts/benchmark_moe.py --router switch
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
contrasts a dense feed-forward model with the MOE version. Pass `--router switch` to use the learned router.

Run it as:

```bash
python scripts/moe_vs_dense.py --router switch
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

- `src/vector_store.py` provides an in-memory `VectorStore` with dot-product
  similarity search.
- `VectorStore.add()` stores embeddings with optional metadata and
  `search()` returns the top-k nearest vectors.
- The store now supports `save()` and `load()` to persist vectors and metadata
  to a compressed `.npz` file.
- This serves as a minimal prototype for the *hierarchical retrieval* memory
  described in the Plan.
- `src/async_vector_store.py` adds `AsyncFaissVectorStore` which wraps the
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

## C-8 Distributed Hierarchical Memory Backend

- Planned extension of `src/hierarchical_memory.py` with an optional gRPC service.
- The store will expose `push_remote()` and `query_remote()` so multiple nodes share one vector database.

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

- The orchestrator in `src/self_play_skill_loop.py` will alternate `self_play_env.rollout_env()` with `robot_skill_transfer.transfer_skills()`.
- Each cycle logs rewards and fine-tunes policies on a small batch of real examples.

## A-9 Automated PR Conflict Checks

- `src/pr_conflict_checker.py` reuses `pull_request_monitor.list_open_prs()` and runs `git merge-base` to detect conflicts.
- Summaries appear in the AutoBench-style scoreboard.

## A-10 Goal-Oriented Evaluation Harness

- `src/eval_harness.py` gathers benchmark metrics from each module and compares them with the targets in `docs/Plan.md`.
- Running `python -m src.eval_harness` prints a pass/fail table for the whole project.

## L-5 Formal Verification Harness

- `src/formal_verifier.py` provides a small property checker that loads model snapshots and symbolically executes critical routines.
- `verify_model()` asserts invariants like gradient norms and output bounds before the model is released.

## M-1 Cross-Modal Fusion Architecture

- `src/cross_modal_fusion.py` embeds text, images and audio in one latent space.
- `train_fusion_model()` fine-tunes a shared encoder-decoder using CLIP- and Whisper-style objectives and `encode_all()` returns embeddings for retrieval.

## M-2 World-Model RL Bridge

- `src/world_model_rl.py` learns a generative world model from logged trajectories and runs model-based RL for rapid policy updates.
- The prototype interfaces with ``gym``-like data and provides ``rollout_policy()`` for offline rollout generation.

## M-3 Self-Calibration for Embodied Agents

- `src/embodied_calibration.py` adapts sensor and actuator parameters from a small set of real-world samples.
- `calibrate()` aligns simulation and hardware spaces so policies trained in simulation remain effective after deployment.

## M-4 Cross-Modal Data Ingestion Pipeline

- `src/data_ingest.py` will align text, image and audio pairs from open datasets.
- Augmentation helpers generate crops and transcripts for training the multi-modal world model.
